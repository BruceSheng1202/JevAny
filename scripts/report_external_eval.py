#!/usr/bin/env python3
"""Score saved predictions, retaining native JevBench and TypeSafe protocols."""
import argparse
from collections import defaultdict
import csv
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
    attempted = [predictions[r["_eval_key"]] for r in records if r["_eval_key"] in predictions]
    def latency(values):
        return {"n": len(values), "median": float(np.median(values)) if values else None,
                "p95": float(np.quantile(values, .95)) if values else None}
    report["latency_ms"] = {
        "model": latency([e["prediction"]["latency_ms"] for e in attempted
                          if "latency_ms" in e.get("prediction", {})]),
        "predictor_wall": latency([e["wall_latency_ms"] for e in attempted if "wall_latency_ms" in e]),
        "reuse": "identical requests reuse the original measured latency",
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
        reference = {
            "provenance": "published_by_Kev_not_rerun", "source_path": path.relative_to(root).as_posix(),
            "source_sha256": digest(path), "matching_panels": matches,
            "match_basis": basis, **evidence,
            "report": report,
        }
        comparison = path.parent / "comparison.json"
        if comparison.exists():
            compared = read_json(comparison)["runs"].get(path.parent.relative_to(root).as_posix())
            if compared is not None:
                reference["typesafe"] = compared
                reference["comparison_sha256"] = digest(comparison)
        references.append(reference)
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


def write_tables(result, destination):
    """Export every model/panel with an explicit metric and completeness column."""
    rows = []
    for model_id, model in result["models"].items():
        for panel_id, panel in model["panels"].items():
            answered = panel["answered_clean"] or {}
            typesafe = panel.get("typesafe", {})
            accuracy, metric = panel["all_requested_accuracy"], "clean knowable question accuracy"
            if "jevbench" in panel:
                accuracy, metric = panel["jevbench"]["accuracy"], "JevBench native public accuracy"
            elif "typesafe" in panel:
                accuracy = panel["typesafe"]["all_rows"]["equal_case_modal_agreement"]
                metric = "TypeSafe equal-case modal agreement"
            rows.append({
                "model": model_id, "panel": panel_id, "complete": panel["complete"],
                "metric": metric, "accuracy": accuracy if panel["complete"] else None,
                "requested_records": panel["requested_records"],
                "evaluated_records": panel["evaluated_records"],
                "rejected_records": len(panel["rejected_records"]),
                "missing_records": len(panel["missing_records"]),
                "answered_clean_questions": answered.get("n"),
                "answered_nll": answered.get("nll"), "answered_brier": answered.get("brier"),
                "answered_ece": answered.get("ece"),
                "answered_coverage_at_5pct_error": answered.get("coverage_at_5pct_error"),
                "typesafe_all_rows_equal_case_tvd": (
                    typesafe.get("all_rows", {}).get("equal_case_total_variation")
                    if panel["complete"] else None),
                "typesafe_answered_equal_case_agreement": (
                    typesafe.get("evaluated", {}).get("equal_case_modal_agreement")),
                "typesafe_answered_equal_case_tvd": (
                    typesafe.get("evaluated", {}).get("equal_case_total_variation")),
                "model_latency_median_ms": panel.get("latency_ms", {}).get("model", {}).get("median"),
                "model_latency_p95_ms": panel.get("latency_ms", {}).get("model", {}).get("p95"),
                "measurement": "local inference", "reference_source": None,
            })
    if not rows:
        return
    empty = dict.fromkeys(rows[0])
    for reference in result.get("official_jev", []):
        report = reference["report"]
        clean = report.get("clean", {})
        coverage = report.get("coverage", {})
        typesafe = reference.get("typesafe", {})
        for panel_id in reference["matching_panels"]:
            accuracy, metric = clean.get("acc"), "published clean question accuracy"
            template = next(iter(result["models"].values()))["panels"][panel_id]
            expected = template["requested_clean_knowable_questions"]
            if expected and clean.get("n") is not None:
                accuracy = clean["acc"] * clean["n"] / expected
                metric = "clean knowable question accuracy; published rejections count wrong"
            if "typesafe" in reference:
                accuracy = reference["typesafe"]["all_rows"]["equal_case_modal_agreement"]
                metric = "TypeSafe equal-case modal agreement"
            rows.append({
                **empty, "model": "jev-gateway-published-by-kev", "panel": panel_id,
                "complete": True, "metric": metric, "accuracy": accuracy,
                "requested_records": coverage.get("requested_records"),
                "evaluated_records": coverage.get("evaluated_records"),
                "rejected_records": coverage.get("rejected_records"),
                "answered_clean_questions": clean.get("n"),
                "answered_nll": clean.get("nll"), "answered_brier": clean.get("brier"),
                "answered_ece": clean.get("ece"),
                "answered_coverage_at_5pct_error": clean.get("coverage_at_5pct_error"),
                "typesafe_all_rows_equal_case_tvd": (
                    typesafe.get("all_rows", {}).get("equal_case_total_variation")),
                "typesafe_answered_equal_case_agreement": (
                    typesafe.get("evaluated", {}).get("equal_case_modal_agreement")),
                "typesafe_answered_equal_case_tvd": (
                    typesafe.get("evaluated", {}).get("equal_case_total_variation")),
                "measurement": reference["provenance"],
                "reference_source": reference["source_path"],
            })
    reference = result.get("official_jev_jevbench")
    if reference:
        for panel_id, panel in reference["panels"].items():
            rows.append({
                **empty, "model": "jev-1.13.0-published-by-jevbench", "panel": panel_id,
                "complete": True, "metric": "JevBench native public accuracy",
                "accuracy": panel["accuracy"], "requested_records": panel["questions"],
                "measurement": reference["provenance"], "reference_source": reference["source_path"],
            })
    with (Path(destination) / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    panel_ids = [panel["id"] for panel in result["panels"]]
    with (Path(destination) / "accuracy.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", *panel_ids])
        by_pair = {(row["model"], row["panel"]): row for row in rows}
        for model in dict.fromkeys(row["model"] for row in rows):
            writer.writerow([model, *[by_pair.get((model, p), {}).get("accuracy") for p in panel_ids]])


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
    write_tables(result, destination)
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
