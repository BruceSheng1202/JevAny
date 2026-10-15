#!/usr/bin/env python3
"""Convert a local AttributionBench split into an evaluation-only JevAny suite."""
import argparse
import json
from pathlib import Path

from jevany.data import materialize
from jevany.datasets.convert import record
from jevany.suite import digest, record_digest, validate_suite, write_json, write_jsonl


SOURCE = "attributionbench"
REPOSITORY = "osunlp/AttributionBench"
LICENSE = "Apache-2.0"
LABELS = {"attributable": True, "not attributable": False}
SPLITS = {"dev": "development", "test": "test", "test_ood": "test"}


def attribution_record(row: dict, *, revision: str, split: str, input_line: int) -> dict:
    """Preserve a claim/reference bundle and its original attribution annotation."""
    if not isinstance(row, dict):
        raise ValueError("each row must be an object")
    for field in ("id", "src_dataset", "claim"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    references = row.get("references")
    if not isinstance(references, list) or any(not isinstance(value, str) for value in references):
        raise ValueError("references must be a list of strings")
    label = row.get("attribution_label")
    if not isinstance(label, str) or label not in LABELS:
        raise ValueError("attribution_label must be attributable or not attributable")
    question = row.get("question")
    if question is not None and not isinstance(question, str):
        raise ValueError("question must be a string or null")

    state = {"claim": row["claim"], "references": list(references)}
    questions = {"support": {
        "type": "noul",
        "instructions": "Is the entire claim supported by the supplied references? "
                        "Judge only the supplied evidence, not outside knowledge.",
        "label": LABELS[label],
        "src": f"{SOURCE}_{row['src_dataset']}",
    }}
    # Related claims/answers to one question must not acquire split-specific groups.
    grouping = {"src_dataset": row["src_dataset"]}
    if question is not None and question.strip():
        grouping["question"] = question
    else:
        grouping["references"] = references
    converted = record(
        SOURCE, row["id"], SPLITS[split], state, questions,
        group_id=f"{SOURCE}/{record_digest(grouping)}",
        repo=REPOSITORY, revision=revision, license=LICENSE,
        original_id=row["id"], src_dataset=row["src_dataset"],
        upstream_split=split, input_line=input_line,
    )
    materialize(converted)
    return converted


def convert_jsonl(
    path: str | Path, output_dir: str | Path, *, revision: str, split: str,
) -> Path:
    """Convert one original split without downloads, relabeling, or overwriting."""
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    if split not in SPLITS:
        raise ValueError("split must be dev, test or test_ood; training is not supported")
    path, output_dir = Path(path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    rows, identifiers = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                converted = attribution_record(
                    json.loads(line), revision=revision, split=split, input_line=line_number,
                )
                identifier = converted["_meta"]["id"]
                if identifier in identifiers:
                    raise ValueError(f"duplicate original id: {converted['_meta']['original_id']}")
                identifiers.add(identifier)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            rows.append(converted)
    if not rows:
        raise ValueError(f"{path}: no records")

    partition = SPLITS[split]
    manifest = {
        "name": f"attributionbench-{split}-v1",
        "base_revisions": {},
        "trainable_sources": [],
        "eval_only_sources": [SOURCE],
        "holdout_sources": [],
        "sources": {SOURCE: {
            "repository": REPOSITORY, "revision": revision, "license": LICENSE,
            "upstream_split": split, "input_file": path.name, "input_sha256": digest(path),
        }},
    }
    output_dir.mkdir(parents=True)
    output = output_dir / f"{partition}.jsonl"
    write_jsonl(output, rows)
    manifest["files"] = {output.name: {
        "records": len(rows), "questions": len(rows), "sha256": digest(output),
    }}
    write_json(output_dir / "manifest.json", manifest)
    validate_suite(output_dir, allow_test=partition == "test")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="local original UTF-8 AttributionBench JSONL export")
    parser.add_argument("--out", required=True, help="new evaluation suite directory")
    parser.add_argument("--revision", required=True, help="upstream revision used to obtain the export")
    parser.add_argument("--split", required=True, choices=tuple(SPLITS), help="original upstream split")
    args = parser.parse_args(argv)
    try:
        output = convert_jsonl(args.path, args.out, revision=args.revision, split=args.split)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
