#!/usr/bin/env python3
"""Apply stricter context limits to completed text evaluations with encoding proof."""
import argparse
from collections import Counter
import copy
import importlib
from pathlib import Path

from jevany.suite import digest, read_json, read_jsonl, record_digest, write_json, write_jsonl


def replay_entry(old: dict, new: dict, prediction: dict, encoder) -> dict:
    """Reuse a prediction only when the ordered request and full encoding match."""
    if record_digest(old["request"]) != record_digest(new["request"]):
        raise ValueError("context replay cannot change the ordered API request")
    if set(old["context"]) != set(new["context"]) or any(
        not 1 <= new["context"][key] <= value for key, value in old["context"].items()
    ):
        raise ValueError("context replay only supports narrower positive limits")
    if old["context"] == new["context"]:
        return copy.deepcopy(prediction)
    proof = {"source_key": old["key"], "source_context": old["context"]}
    try:
        encoded = encoder(new["request"], new["context"])
    except ValueError as error:
        return {"key": new["key"], "status": "rejected",
                "error_type": type(error).__name__, "error": str(error),
                "context_replay": {**proof, "result": "native_context_rejection"}}
    if prediction["status"] != "ok":
        raise ValueError("accepted request has no source prediction; run inference")
    original = encoder(old["request"], old["context"])
    if original != encoded or len(original["ids"]) != prediction["prediction"]["input_tokens"]:
        raise ValueError("source and corrected encodings differ; run inference")
    result = copy.deepcopy(prediction)
    result["key"] = new["key"]
    result["context_replay"] = {
        **proof, "result": "identical_full_encoding", "encoding_sha256": record_digest(encoded),
    }
    return result


def load_encoder(model: dict, metadata: Path, weights: dict):
    """Load only pinned metadata and the native text preprocessor, without GPU weights."""
    import torch

    backend = model["backend"]
    api = importlib.import_module(f"{backend}.api")
    native = importlib.import_module(f"{backend}.model")
    checkpoint = importlib.import_module(f"{backend}.checkpoint")
    head = metadata / model["id"] / "head.pt"
    spec = next((item for item in model.get("weights", []) if item["name"] == "head.pt"), None)
    if spec:
        expected_sha256 = spec["lfs"]["sha256"]
    else:
        from huggingface_hub import hf_hub_download

        repository, separator, pinned_revision = model["checkpoint"].partition("@")
        if not separator or not pinned_revision:
            raise ValueError("context replay requires a pinned checkpoint revision")
        published = hf_hub_download(repository, "head.pt", revision=pinned_revision)
        expected_sha256 = digest(published)
    if digest(head) != expected_sha256:
        raise ValueError(f"{model['id']}: checkpoint metadata checksum mismatch")
    meta = checkpoint.Meta.from_dict(torch.load(head, map_location="cpu", weights_only=True))
    if weights["checkpoint"] != model["checkpoint"] or weights["base"] != meta.base:
        raise ValueError("resolved base provenance differs from checkpoint")
    revision = weights["resolved_base_revision"]
    if not revision or (meta.base_revision and meta.base_revision != revision):
        raise ValueError("resolved base revision differs from checkpoint")
    if backend == "jevany":
        if meta.tokenizer_saved:
            raise ValueError("saved checkpoint tokenizers require fresh inference")
        adapter = meta.backbone_adapter
        if adapter == "auto" and meta.multimodal:
            adapter = checkpoint.get_backbone_adapter(
                "auto", multimodal=True, source=meta.base, revision=revision).name
        tokenizer = native.tokenizer_of(native.load_preprocessor(
            meta.base, revision=revision, multimodal=meta.multimodal, backbone_adapter=adapter))
    else:
        tokenizer = native.load_tokenizer(meta.base, revision=revision)

    def encode(request, context):
        record, _ = api.to_record(api.SystemOneRequest.model_validate(request))
        if record.get("media"):
            raise RuntimeError("context replay supports text requests only")
        for question in record["questions"]:
            question["label"] = 0
        value = native.encode(
            tokenizer, record, max_state=context["max_state"], max_branch=context["max_branch"],
            strict=True, option_isolation=meta.option_isolation)
        if len(value["ids"]) > context["max_packed"]:
            raise ValueError(f"packed request exceeds {context['max_packed']} tokens")
        return value

    return encode, {
        "base": meta.base, "base_revision": revision, "head_sha256": digest(head),
        "option_isolation": meta.option_isolation,
        "native_source_sha256": {name: digest(module.__file__) for name, module in (
            ("api", api), ("model", native), ("checkpoint", checkpoint))},
    }


def load_suite(path: Path):
    manifest = read_json(path / "manifest.json")
    queue = path / manifest["requests"]["path"]
    if digest(queue) != manifest["requests"]["sha256"]:
        raise ValueError(f"{path}: inference queue checksum mismatch")
    requests = read_jsonl(queue)
    if len({item["key"] for item in requests}) != len(requests):
        raise ValueError(f"{path}: duplicate inference keys")
    for item in requests:
        if item["key"] != record_digest({"request": item["request"], "context": item["context"]}):
            raise ValueError(f"{path}: request identity mismatch")
    return manifest, requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-suite", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--weight-provenance", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    old_suite, old_requests = load_suite(args.original_suite)
    suite, requests = load_suite(args.suite)
    if any(old_suite[field] != suite[field] for field in ("models", "sources")):
        raise ValueError("context replay requires the same model and source revisions")
    model = next(item for item in suite["models"] if item["id"] == args.model)
    source = args.runs / model["id"] / "shard-0"
    run = read_json(source / "run.json")
    rows = read_jsonl(source / "predictions.jsonl")
    original = {row["key"]: row for row in rows}
    if (run["model"] != model or run["num_shards"] != 1 or run["shard_index"] != 0
            or run["suite_manifest_sha256"] != digest(args.original_suite / "manifest.json")
            or run["request_queue_sha256"] != old_suite["requests"]["sha256"]):
        raise ValueError("original run provenance differs from its frozen suite")
    if (not read_json(source / "completion.json")["complete"]
            or len(original) != len(rows)
            or set(original) != {item["key"] for item in old_requests}):
        raise ValueError("original model must have complete, unique inference coverage")
    by_key = {item["key"]: item for item in old_requests}
    by_request = {}
    for item in old_requests:
        by_request.setdefault(record_digest(item["request"]), []).append(item)
    weights = next(item for item in read_json(args.weight_provenance)
                   if item["model"] == args.model)
    encoder, encoder_proof = load_encoder(model, args.metadata, weights)
    corrected = []
    for item in requests:
        previous = by_key.get(item["key"])
        if previous is None:
            candidates = by_request.get(record_digest(item["request"]), [])
            previous = next((candidate for candidate in candidates if all(
                candidate["context"][key] >= value for key, value in item["context"].items()
            )), None)
            if previous is None:
                raise ValueError(f"{item['key']}: no source request with sufficient context")
        corrected.append(replay_entry(previous, item, original[previous["key"]], encoder))
    changes = Counter(row["context_replay"]["result"]
                      for row in corrected if "context_replay" in row)
    provenance = {
        **run, "suite_manifest_sha256": digest(args.suite / "manifest.json"),
        "request_queue_sha256": suite["requests"]["sha256"],
        "context_replay": {
            "original_suite_manifest_sha256": run["suite_manifest_sha256"],
            "original_request_queue_sha256": run["request_queue_sha256"],
            "original_run_sha256": digest(source / "run.json"),
            "original_predictions_sha256": digest(source / "predictions.jsonl"),
            "encoder": encoder_proof, "changes": dict(changes),
            "latency": "accepted predictions retain original GPU timing; new rejections have none",
        },
    }
    destination = args.out / model["id"] / "shard-0"
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "run.json", provenance)
    write_jsonl(destination / "predictions.jsonl", corrected)
    write_json(destination / "completion.json", {
        "complete": True, "requests": len(corrected),
        "ok": sum(row["status"] == "ok" for row in corrected),
        "rejected": sum(row["status"] != "ok" for row in corrected),
    })
    print({"model": args.model, "requests": len(corrected), "changes": dict(changes)}, flush=True)


if __name__ == "__main__":
    main()
