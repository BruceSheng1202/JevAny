#!/usr/bin/env python3
"""Freeze the public JevBench and Kev evaluation panels without sampling."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import tarfile
import urllib.request

from jevany.data import api_request, load_records, materialize
from jevany.suite import digest, read_json, read_jsonl, record_digest, write_json, write_jsonl

DEFAULT_RECIPE = Path(__file__).resolve().parents[1] / "recipes/decision-evaluation.json"
SHORT_CONTEXT = {"max_state": 384, "max_branch": 1024, "max_packed": 2048}
PUBLIC_CONTEXT = {"max_state": 8192, "max_branch": 8192, "max_packed": 16384}


def download_sources(recipe, destination):
    """Download pinned public source archives; no account or credentials needed."""
    destination = Path(destination)
    for name, spec in recipe["sources"].items():
        root = destination / name
        marker = root / ".source-revision"
        if marker.exists():
            if marker.read_text().strip() != spec["revision"]:
                raise ValueError(f"{root}: different source revision; use a new directory")
            continue
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"{root}: nonempty source directory without a revision marker")
        root.mkdir(parents=True, exist_ok=True)
        url = f"https://codeload.github.com/{spec['repository']}/tar.gz/{spec['revision']}"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = io.BytesIO(response.read())
        with tarfile.open(fileobj=archive, mode="r:gz") as bundle:
            for member in bundle:
                if not member.isfile():
                    continue
                relative = member.name.split("/", 1)[-1]
                if not (relative.startswith(("evals/", "datasets/", "kev/", "jevbench/",
                                             "runs/jev-", "scripts/compare_typesafe"))
                        or relative == "results/v1.2/jevbench-v1.2-per-task.json"
                        or relative in ("README.md", "LICENSE", "pyproject.toml")):
                    continue
                if not relative.endswith((".json", ".jsonl", ".py", ".md", ".toml")) and relative != "LICENSE":
                    continue
                member.name = relative
                bundle.extract(member, root, filter="data")
        marker.write_text(spec["revision"] + "\n")


def context_limits(value):
    """Return just the three positive encoder limits, excluding descriptive fields."""
    limits = {key: (value or SHORT_CONTEXT)[key] for key in SHORT_CONTEXT}
    if any(type(v) is not int or v < 1 for v in limits.values()):
        raise ValueError("context limits must be positive integers")
    return limits


def request_key(record, context):
    # Insertion order matters: permuting options must produce a different request.
    return record_digest({"request": api_request(record), "context": context_limits(context)})


def jevbench_record(task):
    """Map JevBench labels to System One without modifying state or question text."""
    question = dict(task["question"])
    kind, expected = question["type"], task.get("expected")
    if expected is None or task.get("provenance", {}).get("exclude_reason"):
        raise ValueError(f"{task['id']}: unscorable JevBench item")
    if kind == "noul":
        if task["labels"] != ["no", "yes"] or expected not in ("no", "yes"):
            raise ValueError(f"{task['id']}: invalid JevBench noul labels")
        label = expected == "yes"
    elif kind == "score":
        label = expected
    elif kind == "choice":
        label = expected
    else:
        raise ValueError(f"{task['id']}: unknown question type {kind!r}")
    question.update(label=label, src=task["family"])
    record = {
        "state": task["state"], "questions": {"decision": question},
        "_meta": {"id": task["id"], "group_id": task.get("group") or task["id"],
                  "source": "jevbench", "variant": "clean", "split": "public",
                  "jevbench": task},
    }
    materialize(record)
    return record


def validate_file(path, spec):
    if digest(path) != spec["sha256"]:
        raise ValueError(f"source checksum mismatch: {path}")
    records = read_jsonl(path)
    if len(records) != spec["records"]:
        raise ValueError(f"source record count mismatch: {path}")


def build(recipe, sources, destination, allow_test=False):
    """Write gold-bearing panels and a separate, label-free inference queue."""
    sources, destination = Path(sources), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "panels").mkdir()
    panels, unavailable, excluded, requests = [], [], [], {}

    def add(panel_id, records, source, protocol, context, **metadata):
        context = context_limits(context)
        for record in records:
            key = request_key(record, context)
            record["_eval_key"] = key
            requests.setdefault(key, {"key": key, "request": api_request(record), "context": context})
        filename = f"panels/{panel_id.replace('/', '--')}.jsonl"
        write_jsonl(destination / filename, records)
        panels.append({
            "id": panel_id, "path": filename, "sha256": digest(destination / filename),
            "records": len(records), "questions": sum(len(r["questions"]) for r in records),
            "source": source, "protocol": protocol, "context": context,
            "types": dict(Counter(q["type"] for r in records for q in r["questions"].values())),
            **metadata,
        })

    kev_root = sources / "kev"
    for path in sorted((kev_root / "evals").rglob("manifest.json")):
        manifest = read_json(path)
        relative = path.parent.relative_to(kev_root).as_posix()
        # Some evals/ directories actually contain training/distillation artifacts.
        names = [name for name in ("development.jsonl", "test.jsonl")
                 if manifest.get("files", {}).get(name, {}).get("records", 0)]
        if not names:
            excluded.append({"path": relative, "reason": "no development/test partition"})
        for name in names:
            spec = manifest["files"][name]
            panel_id = "kev/" + relative.removeprefix("evals/") + "/" + name.removesuffix(".jsonl")
            if name == "test.jsonl" and not allow_test:
                excluded.append({"path": panel_id, "reason": "test requires --allow-test"})
                continue
            data = path.parent / name
            metadata = {
                "upstream_path": f"{relative}/{name}", "upstream_sha256": spec["sha256"],
                "upstream_manifest_sha256": digest(path),
                "holdout_sources": manifest.get("holdout_sources", []),
                "split": name.removesuffix(".jsonl"),
            }
            if not data.exists():
                unavailable.append({
                    "id": panel_id, **metadata, "records": spec["records"],
                    "reason": "upstream partition is not public",
                    "mirror": manifest.get("mirror"),
                })
                continue
            validate_file(data, spec)
            protocol = "typesafe" if relative.endswith("typesafe-v1") else "kev"
            add(panel_id, load_records(data, source=relative), "kev", protocol,
                manifest.get("context"), **metadata)

    extra = (
        ("external/ekzhang-mmlupro-v1/records.jsonl", "external/ekzhang-mmlupro-v1/sample.json",
         PUBLIC_CONTEXT, "external test"),
        ("diagnostics/binding-v1.jsonl", "diagnostics/binding-v1.manifest.json",
         SHORT_CONTEXT, "evaluation diagnostic"),
    )
    for filename, manifest_name, context, purpose in extra:
        path = kev_root / "evals" / filename
        meta_path = kev_root / "evals" / manifest_name
        metadata = read_json(meta_path)
        validate_file(path, metadata)
        add("kev/" + filename.removesuffix(".jsonl"), load_records(path, source=filename),
            "kev", "kev", context, split="external", purpose=purpose,
            upstream_path="evals/" + filename, upstream_sha256=digest(path),
            upstream_manifest_sha256=digest(meta_path))

    # These published diagnostic panels were subsequently used for fine-tuning.
    # Keep them visible, but explicitly exclude them from generalization summaries.
    night = kev_root / "evals/night2"
    for name, spec in read_json(night / "manifest.json")["files"].items():
        validate_file(night / name, spec)
        add("kev/night2/" + name.removesuffix(".jsonl"),
            load_records(night / name, source="night2"), "kev", "kev", SHORT_CONTEXT,
            split="training_diagnostic", purpose="later used for Kev fine-tuning",
            upstream_path="evals/night2/" + name, upstream_sha256=spec["sha256"],
            upstream_manifest_sha256=digest(night / "manifest.json"))

    jevbench = sources / "jevbench"
    published = {item["name"]: item for item in read_json(jevbench / "datasets/manifest.json")["splits"]}
    for name in ("easy", "original", "hard"):
        path = jevbench / "datasets/public" / f"{name}.jsonl"
        spec = published[name]
        validate_file(path, {"sha256": spec["sha256"], "records": spec["n"]})
        tasks = read_jsonl(path)
        records = [jevbench_record(task) for task in tasks]
        add(f"jevbench/{name}", records, "jevbench", "jevbench", PUBLIC_CONTEXT,
            split="public", upstream_path=f"datasets/public/{name}.jsonl",
            upstream_sha256=spec["sha256"], public_only=True)

    write_jsonl(destination / "requests.jsonl", list(requests.values()))
    manifest = {
        "version": 1, "sources": recipe["sources"], "models": recipe["models"],
        "test_read_authorized": allow_test, "panels": panels,
        "unavailable": unavailable, "excluded_training_or_empty": excluded,
        "requests": {"path": "requests.jsonl", "sha256": digest(destination / "requests.jsonl"),
                     "unique": len(requests), "logical_records": sum(p["records"] for p in panels)},
        "policy": {
            "sampling": "none", "truncation": "none",
            "reuse": "identical ordered API request and identical context only",
            "labels": "gold and reference distributions are absent from requests.jsonl",
            "jevbench": "public easy/original/hard; no sealed leaderboard composite",
            "selection": "all checkpoints fixed before reading evaluation results",
        },
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    recipe = read_json(args.recipe)
    if args.download:
        download_sources(recipe, args.sources)
    manifest = build(recipe, args.sources, args.out, args.allow_test)
    print(json.dumps({"panels": len(manifest["panels"]), "unavailable": len(manifest["unavailable"]),
                      **manifest["requests"]}, indent=2))


if __name__ == "__main__":
    main()
