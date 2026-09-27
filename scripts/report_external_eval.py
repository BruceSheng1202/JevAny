#!/usr/bin/env python3
"""Score saved predictions, retaining native JevBench and TypeSafe protocols."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from jevany.benchmark import labels, prediction_rows, summarize
from jevany.data import load_records
from jevany.metrics import grouped_metrics, metrics
from jevany.suite import digest, read_json, read_jsonl, write_json


def knowable(source):
    return source != "unknowable" and not source.endswith("_unknowable")


def score_panel(records, predictions):
    """Keep rejected and missing questions in the all-request accuracy denominator."""
    rows, rejected, missing = [], [], []
    for record in records:
        entry = predictions.get(record["_eval_key"])
        if entry is None:
            missing.append(record["_meta"]["id"])
        elif entry["status"] != "ok":
            rejected.append({"id": record["_meta"]["id"], "error": entry.get("error")})
        else:
            rows.extend(prediction_rows(record, entry["prediction"]))
    clean = [r for r in rows if r["variant"] == "clean" and knowable(r["source"])]
    expected = sum(len(r["questions"]) for r in records
                   if r["_meta"]["variant"] == "clean" and knowable(r["_meta"]["source"]))
    correct = sum(int(np.argmax(row["p"]) == row["label"]) for row in clean)
    clean_metrics = metrics(clean) if clean else None
    report = {
        "complete": not missing, "requested_records": len(records),
        "evaluated_records": len(records) - len(rejected) - len(missing),
        "rejected_records": rejected, "missing_records": missing,
        "requested_clean_knowable_questions": expected,
        "all_requested_accuracy": correct / expected if expected else None,
        "answered_clean": clean_metrics,
        "tasks": grouped_metrics(clean, "task") if clean else {},
        "types": grouped_metrics(clean, "type") if clean else {},
        "variants": grouped_metrics(rows, "variant") if rows else {},
        "metric_policy": {
            "failures": "wrong in all_requested_accuracy; no invented probability distribution",
            "missing": "run is incomplete; missing questions remain in the accuracy denominator",
            "calibration": "answered, clean, knowable questions only; shipped temperature",
            "coverage_at_error": "descriptive in-sample threshold, not a deployment guarantee",
            "training_diagnostics": "not an independent generalization score",
        },
    }
    if clean and all("inference_temperature" in r for r in clean):
        temperatures = {r["inference_temperature"] for r in clean}
        if len(temperatures) == 1:
            temperature = next(iter(temperatures))
            report["raw_uncalibrated_clean"] = metrics(clean, temperature=1 / temperature)
    # Preserve the original diagnostic protocol when every record is present.
    # Partial pairs/permutations need the failure-aware report above.
    if rows and not rejected and not missing and any(r["variant"] == "clean" for r in rows):
        upstream_rows = [{**r, "source": "unknowable" if not knowable(r["source"]) else r["source"]}
                         for r in rows]
        if any(knowable(r["source"]) and r["variant"] == "clean" for r in upstream_rows):
            report["decision_diagnostics"] = summarize(upstream_rows)
    unknown = [r for r in rows if not knowable(r["source"])]
    if unknown:
        report["unknowable_confidence"] = {
            "n": len(unknown), "mean_p_max": float(np.mean([max(r["p"]) for r in unknown])),
            "fraction_p_max_ge_0_9": float(np.mean([max(r["p"]) >= .9 for r in unknown])),
            "accuracy": None,
        }
    return report, rows


def typesafe_metrics(records, rows):
    """SemIf/Kev equal-case agreement and TVD, including failures as 0/1."""
    predictions = {row["id"]: dict(zip(row["keys"], row["p"])) for row in rows}
    scores = {}
    for record in records:
        meta = record["_meta"]
        if meta["id"] not in predictions:
            continue
        p, target = predictions[meta["id"]], meta["target"]
        scores[meta["id"]] = (
            float(max(p, key=p.get) == max(target, key=target.get)),
            sum(abs(p[key] - target[key]) for key in target) / 2,
        )

    def aggregate(selected):
        groups = defaultdict(list)
        for record in selected:
            meta = record["_meta"]
            groups[meta["group_id"]].append(scores.get(meta["id"], (0.0, 1.0)))
        return {
            "rows": len(selected), "cases": len(groups),
            "equal_case_modal_agreement": float(np.mean(
                [np.mean([x[0] for x in values]) for values in groups.values()])) if groups else None,
            "equal_case_total_variation": float(np.mean(
                [np.mean([x[1] for x in values]) for values in groups.values()])) if groups else None,
        }
    return {"all_rows": aggregate(records),
            "evaluated": aggregate([r for r in records if r["_meta"]["id"] in scores])}


def jevbench_metrics(records, predictions, model_id):
    """Call the pinned upstream scorer, including its tie and validity rules."""
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize as native_summary
    from jevbench.tasks import Task

    tasks, outputs = [], []
    for record in records:
        task = Task.from_dict(record["_meta"]["jevbench"])
        tasks.append(task)
        entry = predictions.get(record["_eval_key"])
        if entry is None:
            continue
        prediction = entry.get("prediction", {})
        p = prediction.get("probabilities", {}).get("decision", {})
        if task.question["type"] == "noul" and p:
            p = {"no": p["false"], "yes": p["true"]}
        scored = score_task(p, task)
        outputs.append({
            "task_id": task.id, **scored, "ok": entry["status"] == "ok",
            "latency_s": prediction.get("latency_ms", entry.get("wall_latency_ms", 0)) / 1000,
            "model": model_id, "probs_source": "model pointer-head softmax",
            "cost_usd": None, "cost_basis": "not measured",
        })
    return native_summary(tasks, outputs)


def reference_reports(sources, panels):
    """Copy published Jev aggregates without presenting them as a new API run."""
    root = Path(sources) / "kev"
    by_manifest = defaultdict(list)
    for panel in panels:
        if panel.get("upstream_manifest_sha256"):
            by_manifest[panel["upstream_manifest_sha256"]].append(panel["id"])
    references = []
    for path in sorted((root / "runs").glob("jev-*/report.json")):
        report = read_json(path)
        matches = [p for p in by_manifest[report.get("suite_sha256")]
                   if p.endswith("/development") or p.endswith("/records")]
        basis = "original manifest SHA256" if matches else "unmatched original manifest"
        evidence = {}
        if matches:
            identities = {(p["upstream_sha256"], json.dumps(p["context"], sort_keys=True))
                          for p in panels if p["id"] in matches}
            matches = [p["id"] for p in panels
                       if (p["upstream_sha256"], json.dumps(p["context"], sort_keys=True)) in identities]
            basis = "original manifest SHA256; aliases require identical partition SHA256 and context"
        elif report.get("suite") and (path.parent / "rows.json").exists():
            # scienthoon published converted predictions, without a suite hash.
            # Verify all question identities and labels; retain this weaker match
            # explicitly rather than claiming a request-text checksum.
            candidates = [p for p in panels
                          if p.get("upstream_path") == report["suite"] + "/development.jsonl"]
            rows_path = path.parent / "rows.json"
            rows = read_json(rows_path)
            actual = {(r["id"], r["question"]): (r["keys"], r["label"]) for r in rows}
            for panel in candidates:
                records = load_records(root / panel["upstream_path"])
                expected = {(r["_meta"]["id"], qid): labels(q)
                            for r in records for qid, q in r["questions"].items()}
                if len(actual) == len(rows) and expected == actual:
                    matches.append(panel["id"])
                    basis = "source panel path and complete published row IDs, ordered options, labels; no request-text hash"
                    evidence = {"published_rows_sha256": digest(rows_path)}
        references.append({
            "provenance": "published_by_Kev_not_rerun", "source_path": path.relative_to(root).as_posix(),
            "source_sha256": digest(path), "matching_panels": matches,
            "match_basis": basis, **evidence,
            "report": report,
        })
    return references


def jevbench_reference(sources, panel_records):
    """Recover public-tier Jev accuracy from upstream's published per-item outcomes."""
    path = Path(sources) / "jevbench/results/v1.2/jevbench-v1.2-per-task.json"
    if not path.exists():
        return None
    source = read_json(path)
    model = source["systems"]["jev-1.13.0"]
    outcomes = model["public_tasks"]
    result = {
        "provenance": "published_by_JevBench_not_rerun", "model": model["display"],
        "source_path": path.relative_to(Path(sources) / "jevbench").as_posix(),
        "source_sha256": digest(path),
        "match_basis": "public task IDs at the pinned source revision",
        "note": "accuracy only; no probabilities published, so calibration is unavailable",
        "panels": {},
    }
    for ident, records in panel_records.items():
        if not ident.startswith("jevbench/"):
            continue
        ids = [r["_meta"]["id"] for r in records]
        if any(key not in outcomes for key in ids):
            raise ValueError(f"official Jev public outcomes missing from {ident}")
        correct = sum(outcomes[key][0] == "c" for key in ids)
        result["panels"][ident] = {"questions": len(ids), "correct": correct,
                                   "accuracy": correct / len(ids)}
    return result


def collect_predictions(directory, model, suite_hash):
    merged, runs, shard_specs = {}, [], set()
    for path in sorted(Path(directory).glob(f"{model['id']}/**/run.json")):
        metadata = read_json(path)
        if metadata["model"] != model or metadata["suite_manifest_sha256"] != suite_hash:
            raise ValueError(f"{path}: wrong model or suite")
        spec = (metadata["num_shards"], metadata["shard_index"])
        if spec in shard_specs:
            raise ValueError(f"{path}: duplicate shard")
        shard_specs.add(spec)
        runs.append(metadata)
        for entry in read_jsonl(path.parent / "predictions.jsonl"):
            if entry["key"] in merged:
                raise ValueError(f"{path}: duplicate prediction")
            merged[entry["key"]] = entry
    return merged, runs


def report(suite, sources, runs, destination):
    suite, destination = Path(suite), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = read_json(suite / "manifest.json")
    sys.path.insert(0, str(Path(sources).resolve() / "jevbench"))
    result = {
        "suite_manifest_sha256": digest(suite / "manifest.json"),
        "sources": manifest["sources"], "panels": manifest["panels"],
        "unavailable": manifest["unavailable"],
        "official_jev": reference_reports(sources, manifest["panels"]),
        "models": {},
    }
    panel_records = {}
    for panel in manifest["panels"]:
        path = suite / panel["path"]
        if digest(path) != panel["sha256"]:
            raise ValueError(f"panel checksum mismatch: {path}")
        panel_records[panel["id"]] = read_jsonl(path)
    result["official_jev_jevbench"] = jevbench_reference(sources, panel_records)
    for model in manifest["models"]:
        predictions, provenance = collect_predictions(runs, model, result["suite_manifest_sha256"])
        model_reports = {}
        for panel in manifest["panels"]:
            records = panel_records[panel["id"]]
            scored, rows = score_panel(records, predictions)
            if panel["protocol"] == "typesafe":
                scored["typesafe"] = typesafe_metrics(records, rows)
            if panel["protocol"] == "jevbench":
                scored["jevbench"] = jevbench_metrics(records, predictions, model["id"])
            model_reports[panel["id"]] = scored
        result["models"][model["id"]] = {
            "checkpoint": model, "runs": provenance, "panels": model_reports,
        }
        write_json(destination / f"{model['id']}.json", result["models"][model["id"]])
    result["complete"] = all(
        p["complete"] for m in result["models"].values() for p in m["panels"].values())
    write_json(destination / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.suite, args.sources, args.runs, args.out)
    print(json.dumps({"complete": result["complete"], "models": len(result["models"]),
                      "panels": len(result["panels"])}))


if __name__ == "__main__":
    main()
