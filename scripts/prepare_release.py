#!/usr/bin/env python3
"""Stage one portable public JevAny LoRA release without trainer state or private paths."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


ARTIFACTS = (
    "adapter_model.safetensors", "adapter_config.json", "tokenizer.json",
    "tokenizer_config.json", "processor_config.json", "chat_template.jinja",
)
SAFE_ARGUMENTS = (
    "lora_targets", "lora_target_modules", "option_isolation", "special_embeddings",
    "multimodal", "head_dim", "head_residual_dim", "head_type", "query_readout",
    "option_readout", "weights_dtype", "decision_mode",
)
DATASET_SUMMARY = {
    "records": 1_772_725,
    "questions": 2_180_242,
    "categories": [
        "preference", "agent and tool decisions", "reasoning", "classification", "safety",
    ],
}


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inferred_adapter(base):
    lowered = base.lower()
    if "gemma-4" in lowered:
        return "gemma4_vision"
    if "qwen3.5" in lowered or "qwen3.8" in lowered:
        return "qwen_vl"
    if "muse-glimmer" in lowered:
        return "muse_vision"
    raise ValueError(f"pass --backbone-adapter for unrecognized base {base!r}")


def model_card(args, metadata):
    mode = metadata["decision_mode"]
    readout = "direct-token" if mode == "lm_token" else "pointer"
    limit = "255 choices" if mode == "lm_token" else "more than 255 choices (subject to context limits)"
    return f"""---
base_model: {args.base}
library_name: peft
tags:
- jevany
- lora
- decision-model
- {readout}
---

# {args.model_name}

Official JevAny LoRA checkpoint using the **{readout}** readout on
[`{args.base}`](https://huggingface.co/{args.base}). It requires the JevAny code
at the release revision linked from the [project repository](https://github.com/weitianxin/JevAny).

## Evaluation

All values below use the same frozen Transfer and public JevBench protocols.
They are accuracy, not the sealed JevBench leaderboard composite.

| Transfer (1,046) | JevBench Easy (48) | Original (72) | Hard (111) | JevBench total (231) |
|---:|---:|---:|---:|---:|
| {args.transfer_accuracy:.2f}% | {args.jevbench_easy:.2f}% | {args.jevbench_original:.2f}% | {args.jevbench_hard:.2f}% | {args.jevbench_total:.2f}% |

## Training data disclosure

Training used **{DATASET_SUMMARY['records']:,} text records / {DATASET_SUMMARY['questions']:,} labelled decisions**.
It spans preference, agent/tool decisions, reasoning, classification, and safety.

## Readout and limits

This checkpoint uses the **{readout}** readout and supports {limit}. Direct-token
models train with full-vocabulary cross-entropy and are therefore slower to train;
pointer models score option representations with a small learned head. Inference
speed is expected to be similar because both use a single backbone prefill and do
not autoregressively generate an answer.

## Usage

```bash
pip install -e '.[serve,multimodal]'
jevany serve --checkpoint {args.repo_id} --device cuda --dtype bf16
```

The repository contains LoRA adapter weights and JevAny readout metadata, not the
base weights. The base model's license and access terms also apply.
"""


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="source checkpoint directory")
    parser.add_argument("--out", required=True, help="new public checkpoint directory")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--backbone-adapter")
    parser.add_argument("--decision-tokens", required=True,
                        help="comma-separated five-token delimiter alphabet used during training")
    parser.add_argument("--transfer-accuracy", required=True, type=float)
    parser.add_argument("--jevbench-easy", required=True, type=float)
    parser.add_argument("--jevbench-original", required=True, type=float)
    parser.add_argument("--jevbench-hard", required=True, type=float)
    args = parser.parse_args(argv)
    args.jevbench_total = (
        48 * args.jevbench_easy + 72 * args.jevbench_original + 111 * args.jevbench_hard
    ) / 231

    source, output = Path(args.run), Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    required = ("adapter_model.safetensors", "adapter_config.json", "head.pt",
                "tokenizer.json", "tokenizer_config.json")
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    decision_tokens = args.decision_tokens.split(",")
    if len(decision_tokens) != 5 or len(set(decision_tokens)) != 5:
        raise ValueError("--decision-tokens must contain five distinct tokens")
    backbone_adapter = args.backbone_adapter or inferred_adapter(args.base)
    output.mkdir(parents=True)
    for name in ARTIFACTS:
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)

    config_path = output / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = args.base
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tokenizer_path = output / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_config.pop("is_local", None)
    tokenizer_config.pop("local_files_only", None)
    tokenizer_config["jevany_decision_tokens"] = decision_tokens
    tokenizer_path.write_text(json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    original = torch.load(source / "head.pt", map_location="cpu", weights_only=False)
    old_args = original.get("args") or {}
    mode = original.get("decision_mode") or "pointer"
    metadata = {
        "base": args.base,
        "base_revision": original.get("base_revision"),
        "lora": original.get("lora", config.get("r", 0)),
        "head": original.get("head"),
        "head_dim": original.get("head_dim", 256),
        "head_residual_dim": original.get("head_residual_dim", 0),
        "head_type": original.get("head_type", "linear"),
        "query_readout": original.get("query_readout", "decide"),
        "option_readout": original.get("option_readout", "close"),
        "option_isolation": bool(original.get("option_isolation", False)),
        "special_embeddings": bool(original.get("special_embeddings", False)),
        "multimodal": bool(original.get("multimodal", False)),
        "backbone_adapter": backbone_adapter,
        "branch_mode": "rows",
        "tokenizer_saved": True,
        "weights_dtype": original.get("weights_dtype", "fp32"),
        "decision_mode": mode,
        "verbalizers": original.get("verbalizers") or [],
        "temperature": float(original.get("temperature", 1.0)),
        "holdout": [],
        "args": {key: old_args[key] for key in SAFE_ARGUMENTS if key in old_args},
        "training_data": DATASET_SUMMARY,
        "evaluation": {
            "transfer_v9": {"questions": 1046, "accuracy": args.transfer_accuracy / 100},
            "jevbench_public": {
                "questions": 231,
                "accuracy": args.jevbench_total / 100,
                "tiers": {"easy": args.jevbench_easy / 100,
                          "original": args.jevbench_original / 100,
                          "hard": args.jevbench_hard / 100},
            },
        },
    }
    metadata["args"].update({
        "base": args.base,
        "base_revision": metadata["base_revision"],
        "lora": metadata["lora"],
        "head_dim": metadata["head_dim"],
        "head_residual_dim": metadata["head_residual_dim"],
        "head_type": metadata["head_type"],
        "query_readout": metadata["query_readout"],
        "option_readout": metadata["option_readout"],
        "multimodal": metadata["multimodal"],
        "weights_dtype": metadata["weights_dtype"],
        "decision_mode": mode,
    })
    torch.save(metadata, output / "head.pt")
    (output / "README.md").write_text(model_card(args, metadata), encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    for name in ("LICENSE", "NOTICE", "ACKNOWLEDGEMENTS.md"):
        shutil.copy2(root / name, output / name)

    serialized = json.dumps({key: value for key, value in metadata.items() if key != "head"}, default=str)
    public_text = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir()
                            if path.suffix in {".md", ".json", ".jinja"})
    forbidden = ("/lustre", "/home/", "weakness-targeted", "four-backbone", "intern-decision",
                 "source-evaluation", "license_pending")
    found = [value for value in forbidden if value.casefold() in (serialized + public_text).casefold()]
    if found:
        raise ValueError(f"private metadata remains: {found}")

    manifest = {
        "format": "JevAny checkpoint v2",
        "repository": args.repo_id,
        "base": args.base,
        "base_revision": metadata["base_revision"],
        "decision_mode": mode,
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.name != "release-manifest.json":
            manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (output / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"prepared {output}")


if __name__ == "__main__":
    main()
