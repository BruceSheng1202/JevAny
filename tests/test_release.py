"""Release staging keeps required weights while removing private provenance."""
import json
import re
import tomllib
from pathlib import Path

import torch

from scripts.prepare_release import main
from scripts.verify_release import main as verify_release, sha256


def test_release_metadata_uses_org_and_current_version():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    citation = (root / "CITATION.cff").read_text()
    package = (root / "jevany" / "__init__.py").read_text()
    release_script = (root / "scripts" / "prepare_release.py").read_text()

    version = project["project"]["version"]
    assert project["project"]["urls"]["Repository"] == "https://github.com/SimpleJev/JevAny"
    assert "repository-code: https://github.com/SimpleJev/JevAny" in citation
    assert re.search(rf"^version: {re.escape(version)}$", citation, re.MULTILINE)
    assert re.search(rf'^__version__ = "{re.escape(version)}"$', package, re.MULTILINE)
    assert "https://github.com/SimpleJev/JevAny" in release_script

    current_metadata = "\n".join((citation, package, release_script))
    assert "github.com/weitianxin/JevAny" not in current_metadata
    assert "github.com/tianxinwei/JevAny" not in current_metadata


def test_prepare_release_sanitizes_checkpoint(tmp_path):
    source, output = tmp_path / "private", tmp_path / "public"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    (source / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "/lustre/private/base", "r": 8,
        "lora_dropout": 0.0, "target_modules": ["q_proj"],
    }))
    (source / "tokenizer.json").write_text("{}")
    (source / "tokenizer_config.json").write_text(json.dumps({
        "is_local": True, "local_files_only": True, "tokenizer_class": "FakeTokenizer",
    }))
    (source / "trainer_state.pt").write_bytes(b"private optimizer")
    torch.save({
        "base": "/lustre/private/base", "base_revision": "abc", "lora": 8,
        "head": {"q.weight": torch.ones(1)}, "head_dim": 256,
        "head_residual_dim": 256, "head_type": "residual", "multimodal": True,
        "weights_dtype": "bf16", "args": {
            "lora_targets": "all", "data": "/lustre/private/data.jsonl",
            "out": "/lustre/private/run",
        },
    }, source / "head.pt")

    main([
        "--run", str(source), "--out", str(output),
        "--repo-id", "owner/model", "--model-name", "Model",
        "--base", "Qwen/Qwen3.5-4B",
        "--decision-tokens", "<a>,<b>,<c>,<d>,<e>",
        "--transfer-accuracy", "80", "--jevbench-easy", "100",
        "--jevbench-original", "90", "--jevbench-hard", "60",
    ])

    metadata = torch.load(output / "head.pt", map_location="cpu", weights_only=False)
    tokenizer = json.loads((output / "tokenizer_config.json").read_text())
    adapter = json.loads((output / "adapter_config.json").read_text())
    assert metadata["base"] == adapter["base_model_name_or_path"] == "Qwen/Qwen3.5-4B"
    assert metadata["backbone_adapter"] == "qwen_vl"
    assert metadata["branch_mode"] == "rows" and metadata["tokenizer_saved"]
    assert metadata["args"] == {
        "lora_targets": "all", "base": "Qwen/Qwen3.5-4B", "base_revision": "abc",
        "lora": 8, "head_dim": 256, "head_residual_dim": 256,
        "head_type": "residual", "query_readout": "decide", "option_readout": "close",
        "multimodal": True, "weights_dtype": "bf16", "decision_mode": "pointer",
    }
    assert tokenizer["jevany_decision_tokens"] == ["<a>", "<b>", "<c>", "<d>", "<e>"]
    assert "is_local" not in tokenizer and "local_files_only" not in tokenizer
    assert not (output / "trainer_state.pt").exists()
    public = "\n".join(path.read_text(errors="ignore") for path in output.iterdir()
                       if path.suffix in {".json", ".md"})
    assert "/lustre" not in public


def test_verify_release_rejects_score_drift(tmp_path):
    source, output = tmp_path / "private", tmp_path / "release" / "Model"
    source.mkdir(parents=True)
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    (source / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "old", "r": 8, "target_modules": ["q_proj"],
    }))
    (source / "tokenizer.json").write_text("{}")
    (source / "tokenizer_config.json").write_text("{}")
    torch.save({"base_revision": "abc", "head": {}}, source / "head.pt")
    main([
        "--run", str(source), "--out", str(output), "--repo-id", "owner/Model",
        "--model-name", "Model", "--base", "Qwen/Qwen3.5-4B",
        "--decision-tokens", "<a>,<b>,<c>,<d>,<e>",
        "--transfer-accuracy", "80", "--jevbench-easy", "100",
        "--jevbench-original", "90", "--jevbench-hard", "60",
    ])
    results = {
        "training_data": {
            "records": 1772725, "questions": 2180242,
            "categories": ["preference", "agent and tool decisions", "reasoning", "classification", "safety"],
        },
        "protocols": {"jevbench_public": {
            "claim": "public development diagnostic; not a sealed Benchmark Heaven score",
            "questions": 231, "suite": "jevbench-public-v1.4.2.2",
            "suite_sha256": "b6d6eb34fdbbf46ec11be8657da5e513e3538990d7a00ca5582c55d8bd7941c6",
        }},
        "released_models": [{
            "repository": "owner/Model", "readout": "pointer", "transfer_v9_accuracy": 0.8,
            "jevbench_public_accuracy": (48 + 72 * .9 + 111 * .6) / 231,
            "jevbench_tiers": {"easy": 1.0, "original": .9, "hard": .6},
        }],
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))
    verify_release(["--release-dir", str(output.parent), "--results", str(results_path)])
    metadata = torch.load(output / "head.pt", map_location="cpu", weights_only=False)
    metadata["evaluation"]["jevbench_public"]["accuracy"] = 0.5
    torch.save(metadata, output / "head.pt")
    manifest_path = output / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["head.pt"] = {
        "bytes": (output / "head.pt").stat().st_size,
        "sha256": sha256(output / "head.pt"),
    }
    manifest_path.write_text(json.dumps(manifest))
    try:
        verify_release(["--release-dir", str(output.parent), "--results", str(results_path)])
    except ValueError as error:
        assert "JevBench total mismatch" in str(error)
    else:
        raise AssertionError("release verification accepted changed metrics")
