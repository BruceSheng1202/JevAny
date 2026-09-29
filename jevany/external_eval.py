"""Resumable inference for frozen external decision panels.

The queue contains API requests without gold labels. Each process owns one GPU
and one output file; scoring happens separately after inference.
"""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import time

from jevany.api import question_keys, validate_distribution
from jevany.suite import digest, read_json, read_jsonl, write_json


class DecisionPredictor:
    """Run JevAny or the pinned upstream Kev implementation on the same requests."""

    def __init__(self, backend, checkpoint, device="cuda", dtype=None):
        import torch

        if backend not in ("jevany", "kev"):
            raise ValueError("backend must be jevany or kev")
        self.backend, self.device = backend, device
        self.api = importlib.import_module(f"{backend}.api")
        module = importlib.import_module(f"{backend}.checkpoint")
        options = module.LoadOptions(dtype={"fp32": torch.float32, "bf16": torch.bfloat16}.get(dtype))
        self.checkpoint = module.Checkpoint(checkpoint)
        if str(device).startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.tokenizer, self.model = self.checkpoint.load(device, options)
        self.temperature = getattr(
            self.model, "temperature",
            self.model.head.temperature if self.model.head is not None else 1.0,
        )
        self.provenance = {
            "backend": backend, "requested_checkpoint": checkpoint,
            "base": self.checkpoint.meta.base,
            "base_revision": self.checkpoint.meta.base_revision,
            "temperature": self.temperature,
            "weights_dtype": str(next(self.model.lm.parameters()).dtype),
            "requested_dtype": dtype or "checkpoint default",
            "attention": "math; TF32 disabled", "device": device,
        }

    def __call__(self, request, context):
        import torch

        clean = self.api.SystemOneRequest.model_validate(request)
        record, metadata = self.api.to_record(clean)
        # Both encoders carry a label slot for training. Its value never enters
        # tokens or inference; fill a constant, not the evaluation answer.
        for question in record["questions"]:
            question["label"] = 0
        with torch.inference_mode():
            encoded = self.model.encode(
                self.tokenizer, record, max_state=context["max_state"],
                max_branch=context["max_branch"], strict=True,
            )
            if len(encoded["ids"]) > context["max_packed"]:
                raise ValueError(f"packed request exceeds {context['max_packed']} tokens")
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = self.model.forward(encoded)
            probabilities = [torch.softmax(z, -1).cpu().tolist() for z in logits]
            values = [z.float().cpu() for z in logits]
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            latency = (time.perf_counter() - start) * 1000
        keys = [item["keys"] for item in metadata]
        ids = list(request["questions"])
        return {
            "probabilities": {qid: dict(zip(ks, p)) for qid, ks, p in zip(ids, keys, probabilities)},
            "logits": {qid: dict(zip(ks, z.tolist())) for qid, ks, z in zip(ids, keys, values)},
            "inference_temperature": self.temperature, "latency_ms": latency,
            "input_tokens": len(encoded["ids"]),
        }


def load_completed(path, expected_keys):
    """Recover complete JSONL entries, discarding only a torn final write."""
    path = Path(path)
    if not path.exists():
        return {}
    completed, offset = {}, 0
    with path.open("rb") as stream:
        while line := stream.readline():
            if not line.endswith(b"\n"):
                break
            entry = json.loads(line)
            if entry["key"] not in expected_keys or entry["key"] in completed:
                raise ValueError(f"{path}: duplicate or unexpected request key")
            completed[entry["key"]] = entry
            offset = stream.tell()
    if offset != path.stat().st_size:
        with path.open("r+b") as stream:
            stream.truncate(offset)
    return completed


def run_requests(requests, predictor, output, provenance):
    """Record every success or rejection; authentication failures abort immediately."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run.json"
    if manifest_path.exists():
        if read_json(manifest_path) != provenance:
            raise ValueError(f"{output}: resume configuration differs")
    else:
        write_json(manifest_path, provenance)
    keys = {r["key"] for r in requests}
    if len(keys) != len(requests):
        raise ValueError("duplicate inference request keys")
    completed = load_completed(output / "predictions.jsonl", keys)
    with (output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
        for index, item in enumerate(requests):
            if item["key"] in completed:
                continue
            started = time.perf_counter()
            try:
                prediction = predictor(copy.deepcopy(item["request"]), item["context"])
                if set(prediction["probabilities"]) != set(item["request"]["questions"]):
                    raise ValueError("prediction question IDs differ from request")
                for qid, question in item["request"]["questions"].items():
                    validate_distribution(prediction["probabilities"][qid],
                                          question_keys(question["type"], question.get("criteria")))
                entry = {"key": item["key"], "status": "ok", "prediction": prediction}
            except Exception as error:
                import torch
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                elif not isinstance(error, ValueError):
                    write_json(output / "failure.json", {
                        "key": item["key"], "error_type": type(error).__name__,
                        "error": str(error),
                    })
                    raise
                entry = {"key": item["key"], "status": "rejected",
                         "error_type": type(error).__name__, "error": str(error)}
            entry["wall_latency_ms"] = (time.perf_counter() - started) * 1000
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
            stream.flush()
            completed[item["key"]] = entry
            if (index + 1) % 100 == 0:
                print(json.dumps({"completed": index + 1, "total": len(requests)}), flush=True)
    result = {"complete": len(completed) == len(requests), "requests": len(requests),
              "ok": sum(e["status"] == "ok" for e in completed.values()),
              "rejected": sum(e["status"] != "ok" for e in completed.values())}
    write_json(output / "completion.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model", required=True, help="model id in the frozen suite manifest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    suite = read_json(args.suite / "manifest.json")
    path = args.suite / suite["requests"]["path"]
    if digest(path) != suite["requests"]["sha256"]:
        raise ValueError("inference queue checksum mismatch")
    model = next((m for m in suite["models"] if m["id"] == args.model), None)
    if model is None:
        parser.error(f"unknown frozen model: {args.model}")
    if args.device == "cuda":
        import torch
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    requests = read_jsonl(path)[args.shard_index::args.num_shards]
    predictor = DecisionPredictor(model["backend"], model["checkpoint"], args.device, args.dtype)
    provenance = {
        "model": model, "suite_manifest_sha256": digest(args.suite / "manifest.json"),
        "request_queue_sha256": digest(path), "sources": suite["sources"],
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "runtime": predictor.provenance,
    }
    print(json.dumps(provenance), flush=True)
    print(json.dumps(run_requests(requests, predictor, args.out, provenance)), flush=True)


if __name__ == "__main__":
    main()
