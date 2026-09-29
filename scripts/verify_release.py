#!/usr/bin/env python3
"""Fail closed when staged model artifacts disagree with release results."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


EXPECTED_DATA = {
    "records": 1_772_725,
    "questions": 2_180_242,
    "categories": [
        "preference", "agent and tool decisions", "reasoning", "classification", "safety",
    ],
}
FORBIDDEN = (
    "/lustre", "/home/", "weakness-targeted", "four-backbone", "intern-decision",
    "source-evaluation", "license_pending",
)


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def close(left, right, tolerance=5e-10):
    return abs(float(left) - float(right)) <= tolerance


def verify_directory(directory, expected):
    manifest_path = directory / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["repository"] != expected["repository"]:
        raise ValueError(f"repository mismatch in {directory}")
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    expected_files = set(manifest["files"]) | {manifest_path.name}
    if actual_files != expected_files:
        raise ValueError(f"untracked or missing release files in {directory}")
    for name, record in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"manifest mismatch: {path}")

    metadata = torch.load(directory / "head.pt", map_location="cpu", weights_only=False)
    if metadata["decision_mode"] != expected["readout"].replace("direct-token", "lm_token"):
        raise ValueError(f"readout mismatch in {directory}")
    if metadata.get("training_data") != EXPECTED_DATA:
        raise ValueError(f"training-data disclosure mismatch in {directory}")
    evaluation = metadata["evaluation"]
    if not close(evaluation["transfer_v9"]["accuracy"], expected["transfer_v9_accuracy"]):
        raise ValueError(f"Transfer-v9 mismatch in {directory}")
    jevbench = evaluation["jevbench_public"]
    if not close(jevbench["accuracy"], expected["jevbench_public_accuracy"]):
        raise ValueError(f"JevBench total mismatch in {directory}")
    for tier, accuracy in expected["jevbench_tiers"].items():
        if not close(jevbench["tiers"][tier], accuracy):
            raise ValueError(f"JevBench {tier} mismatch in {directory}")
    weighted = (
        48 * jevbench["tiers"]["easy"]
        + 72 * jevbench["tiers"]["original"]
        + 111 * jevbench["tiers"]["hard"]
    ) / 231
    if not close(weighted, jevbench["accuracy"]):
        raise ValueError(f"JevBench tier/total mismatch in {directory}")

    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in directory.iterdir() if path.suffix in {".md", ".json", ".jinja"}
    )
    leaked = [needle for needle in FORBIDDEN if needle.casefold() in public_text.casefold()]
    if leaked:
        raise ValueError(f"private release text in {directory}: {leaked}")
    for accuracy in (expected["transfer_v9_accuracy"], expected["jevbench_public_accuracy"]):
        if f"{100 * accuracy:.2f}%" not in (directory / "README.md").read_text(encoding="utf-8"):
            raise ValueError(f"model card score mismatch in {directory}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--results", default="results/model-family-v2.json")
    args = parser.parse_args(argv)
    release_dir = Path(args.release_dir)
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if results["training_data"] != EXPECTED_DATA:
        raise ValueError("unexpected aggregate training-data disclosure")
    protocol = results["protocols"]["jevbench_public"]
    if protocol != {
        "claim": "public development diagnostic; not a sealed Benchmark Heaven score",
        "questions": 231,
        "suite": "jevbench-public-v1.4.2.2",
        "suite_sha256": "b6d6eb34fdbbf46ec11be8657da5e513e3538990d7a00ca5582c55d8bd7941c6",
    }:
        raise ValueError("JevBench protocol is not the frozen canonical suite")
    expected = {item["repository"].split("/", 1)[1]: item for item in results["released_models"]}
    actual = {path.name for path in release_dir.iterdir() if path.is_dir()}
    if actual != set(expected):
        raise ValueError("staged model set does not match release results")
    for name, record in expected.items():
        verify_directory(release_dir / name, record)
    print(f"verified {len(expected)} release directories")


if __name__ == "__main__":
    main()
