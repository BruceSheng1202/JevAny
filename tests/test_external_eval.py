"""Protocol and recovery checks without model weights, accounts, or network."""
import json

import pytest

from jevany.external_eval import load_completed, run_requests
from scripts.build_external_eval import (
    PUBLIC_CONTEXT, SHORT_CONTEXT, build, jevbench_record, request_key, validate_file,
)
from scripts.replay_external_context import replay_entry
from scripts.report_external_eval import (
    jevbench_reference, reference_reports, score_panel, typesafe_metrics, write_tables,
)


def task(kind="noul", expected="yes"):
    return {
        "id": "item", "family": "policy", "state": {"evidence": "A"},
        "question": {"type": kind, "instructions": "Choose",
                     **({"criteria": ["low", "high"]} if kind == "score" else {})},
        "labels": ["0", "1"] if kind == "score" else ["no", "yes"],
        "expected": expected, "split": "public", "provenance": {},
    }


def test_mapping_and_inference_identity_keep_gold_out():
    a, b = jevbench_record(task(expected="yes")), jevbench_record(task(expected="no"))
    assert a["questions"]["decision"]["label"] is True
    assert b["questions"]["decision"]["label"] is False
    assert request_key(a, PUBLIC_CONTEXT) == request_key(b, PUBLIC_CONTEXT)
    assert jevbench_record(task("score", 1))["questions"]["decision"]["label"] == 1
    with pytest.raises(ValueError, match="invalid JevBench noul"):
        jevbench_record(task(expected="true"))
    with pytest.raises(ValueError, match="integer"):
        jevbench_record(task("score", "1"))


def test_request_identity_preserves_option_order_and_context():
    a = {"state": "x", "questions": {"q": {
        "type": "choice", "criteria": {"a": "A", "b": "B"}, "label": "a"}}}
    b = {"state": "x", "questions": {"q": {
        "type": "choice", "criteria": {"b": "B", "a": "A"}, "label": "a"}}}
    assert request_key(a, PUBLIC_CONTEXT) != request_key(b, PUBLIC_CONTEXT)
    assert request_key(a, PUBLIC_CONTEXT) != request_key(a, {**PUBLIC_CONTEXT, "max_state": 4096})


def test_invalid_source_does_not_pass_freeze(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        validate_file(path, {"sha256": "wrong", "records": 1})


def test_record_only_mmlu_uses_native_default_context(tmp_path):
    from jevany.suite import digest, write_json, write_jsonl

    sources = tmp_path / "sources"
    record = jevbench_record(task())
    for filename, manifest in (
        ("external/ekzhang-mmlupro-v1/records.jsonl", "external/ekzhang-mmlupro-v1/sample.json"),
        ("diagnostics/binding-v1.jsonl", "diagnostics/binding-v1.manifest.json"),
    ):
        path = sources / "kev/evals" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, [record])
        write_json(sources / "kev/evals" / manifest, {"sha256": digest(path), "records": 1})
    night = sources / "kev/evals/night2"
    night.mkdir()
    write_json(night / "manifest.json", {"files": {}})
    public = sources / "jevbench/datasets/public"
    public.mkdir(parents=True)
    splits = []
    for name in ("easy", "original", "hard"):
        path = public / f"{name}.jsonl"
        write_jsonl(path, [task()])
        splits.append({"name": name, "sha256": digest(path), "n": 1})
    write_json(public.parent / "manifest.json", {"splits": splits})
    suite = build({"models": [], "sources": {}}, sources, tmp_path / "suite")
    mmlu = next(panel for panel in suite["panels"] if "mmlupro" in panel["id"])
    assert mmlu["context"] == SHORT_CONTEXT
    assert all(panel["context"] == PUBLIC_CONTEXT for panel in suite["panels"]
               if panel["source"] == "jevbench")


def test_context_replay_requires_identical_encoding_and_order():
    request = {"state": "x", "questions": {"q": {
        "type": "choice", "criteria": {"a": "A", "b": "B"}}}}
    old = {"key": "old", "request": request, "context": PUBLIC_CONTEXT}
    new = {**old, "key": "new", "context": SHORT_CONTEXT}
    source = {"key": "old", "status": "ok", "wall_latency_ms": 4,
              "prediction": {"input_tokens": 2, "latency_ms": 3}}
    encoded = {"ids": [1, 2], "pos": [0, 1]}
    result = replay_entry(old, new, source, lambda *_: encoded)
    assert result["key"] == "new" and source["key"] == "old"
    assert result["prediction"] == source["prediction"]
    assert result["context_replay"]["result"] == "identical_full_encoding"
    with pytest.raises(ValueError, match="encodings differ"):
        replay_entry(old, new, source, lambda _, context: {
            **encoded, "pos": [0, int(context == PUBLIC_CONTEXT)]})
    permuted = {**request, "questions": {"q": {
        "type": "choice", "criteria": {"b": "B", "a": "A"}}}}
    with pytest.raises(ValueError, match="ordered API request"):
        replay_entry(old, {**new, "request": permuted}, source, lambda *_: encoded)
    with pytest.raises(ValueError, match="narrower"):
        replay_entry(new, old, source, lambda *_: encoded)


def test_context_replay_rejection_has_no_reused_prediction_or_latency():
    def overlong(*_):
        raise ValueError("state exceeds 384 tokens")

    old = {"key": "old", "request": {}, "context": PUBLIC_CONTEXT}
    new = {**old, "key": "new", "context": SHORT_CONTEXT}
    result = replay_entry(old, new, {"status": "ok", "wall_latency_ms": 4}, overlong)
    assert result["status"] == "rejected"
    assert "prediction" not in result and "wall_latency_ms" not in result


def test_resume_only_repairs_torn_final_line(tmp_path):
    path = tmp_path / "predictions.jsonl"
    good = json.dumps({"key": "a", "status": "ok"}) + "\n"
    path.write_text(good + '{"key":')
    assert set(load_completed(path, {"a", "b"})) == {"a"}
    assert path.read_text() == good
    path.write_text(good + good)
    with pytest.raises(ValueError, match="duplicate"):
        load_completed(path, {"a"})


def test_resume_and_rejection_keep_all_requests(tmp_path):
    request = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "yes?"}}}
    requests = [{"key": "a", "request": request, "context": PUBLIC_CONTEXT},
                {"key": "b", "request": {**request, "state": "long"}, "context": PUBLIC_CONTEXT}]
    calls = []

    def predictor(req, context):
        assert "label" not in req["questions"]["q"]
        calls.append(req["state"])
        if req["state"] == "long":
            raise ValueError("state exceeds context")
        return {"probabilities": {"q": {"false": .1, "true": .9}}}

    result = run_requests(requests, predictor, tmp_path, {"model": "fixture"})
    assert result == {"complete": True, "requests": 2, "ok": 1, "rejected": 1}
    assert run_requests(requests, predictor, tmp_path, {"model": "fixture"}) == result
    assert len(calls) == 2
    with pytest.raises(ValueError, match="resume configuration"):
        run_requests(requests, predictor, tmp_path, {"model": "other"})


def test_rejected_and_missing_are_wrong_without_fake_probabilities():
    records = []
    for key in ("a", "b", "c"):
        record = jevbench_record({**task(), "id": key})
        record["_eval_key"] = key
        records.append(record)
    predictions = {
        "a": {"status": "ok", "prediction": {
            "probabilities": {"decision": {"false": .1, "true": .9}}}},
        "b": {"status": "rejected", "error": "context"},
    }
    scored, rows = score_panel(records, predictions)
    assert scored["all_requested_accuracy"] == pytest.approx(1 / 3)
    assert scored["answered_clean"]["acc"] == 1
    assert scored["answered_clean"]["n"] == len(rows) == 1
    assert not scored["complete"] and scored["missing_records"] == ["c"]


def test_unknown_evidence_is_not_scored_for_accuracy():
    record = jevbench_record(task())
    record["_meta"]["source"] = "night2_unknowable"
    record["_eval_key"] = "a"
    scored, _ = score_panel([record], {
        "a": {"status": "ok", "prediction": {
            "probabilities": {"decision": {"false": .1, "true": .9}}}},
    })
    assert scored["all_requested_accuracy"] is None
    assert scored["unknowable_confidence"]["fraction_p_max_ge_0_9"] == 1


def test_typesafe_weights_cases_equally_and_penalizes_missing(tmp_path):
    import csv

    records, rows = [], []
    for ident, group in (("a", "case1"), ("b", "case1"), ("c", "case2")):
        record = jevbench_record({**task(), "id": ident, "group": group})
        record["_meta"]["target"] = {"false": 0.0, "true": 1.0}
        record["_eval_key"] = ident
        records.append(record)
        if ident != "c":
            rows.append({"id": ident, "keys": ["false", "true"], "p": [0.0, 1.0]})
    result = typesafe_metrics(records, rows)
    assert result["evaluated"]["equal_case_modal_agreement"] == 1
    assert result["all_rows"]["equal_case_modal_agreement"] == .5
    assert result["all_rows"]["equal_case_total_variation"] == .5
    predictions = {ident: {"status": "ok", "prediction": {
        "probabilities": {"decision": {"false": 0.0, "true": 1.0}}}}
        for ident in ("a", "b")}
    predictions["c"] = {"status": "rejected", "error": "context"}
    panel, _ = score_panel(records, predictions)
    panel["typesafe"] = result
    assert panel["all_requested_accuracy"] == pytest.approx(2 / 3)
    write_tables({"panels": [{"id": "typesafe"}], "models": {
        "local": {"panels": {"typesafe": panel}}}}, tmp_path)
    with (tmp_path / "scores.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["accuracy"] == row["typesafe_all_rows_equal_case_tvd"] == "0.5"
    assert row["typesafe_answered_equal_case_agreement"] == "1.0"
    assert row["typesafe_answered_equal_case_tvd"] == "0.0"


def test_published_reference_alias_requires_identical_data_and_context(tmp_path):
    source = tmp_path / "kev/runs/jev-fixture"
    source.mkdir(parents=True)
    (source / "report.json").write_text(json.dumps({"suite_sha256": "manifest"}))
    panel = {"id": "kev/original/development", "upstream_manifest_sha256": "manifest",
             "upstream_sha256": "data", "context": PUBLIC_CONTEXT}
    alias = {**panel, "id": "kev/alias/development", "upstream_manifest_sha256": "other"}
    changed = {**alias, "id": "kev/changed/development", "upstream_sha256": "different"}
    reference = reference_reports(tmp_path, [panel, alias, changed])[0]
    assert reference["matching_panels"] == [panel["id"], alias["id"]]
    assert reference["provenance"] == "published_by_Kev_not_rerun"


def test_jevbench_reference_uses_only_exact_public_ids(tmp_path):
    source = tmp_path / "jevbench/results/v1.2"
    source.mkdir(parents=True)
    (source / "jevbench-v1.2-per-task.json").write_text(json.dumps({
        "systems": {"jev-1.13.0": {"display": "Jev 1.13.0", "public_tasks": {"item": ["c", .3]}}},
    }))
    records = {"jevbench/easy": [jevbench_record(task())]}
    reference = jevbench_reference(tmp_path, records)
    assert reference["panels"]["jevbench/easy"]["accuracy"] == 1
    assert "brier" not in reference["panels"]["jevbench/easy"]
    records["jevbench/easy"][0]["_meta"]["id"] = "unknown"
    with pytest.raises(ValueError, match="missing"):
        jevbench_reference(tmp_path, records)


def test_table_uses_native_metric_and_leaves_incomplete_accuracy_blank(tmp_path):
    import csv

    record = jevbench_record(task())
    record["_eval_key"] = "a"
    incomplete, _ = score_panel([record], {})
    native = {**incomplete, "complete": True, "missing_records": [],
              "jevbench": {"accuracy": .75}}
    result = {"panels": [{"id": "p"}], "models": {
        "missing": {"panels": {"p": incomplete}}, "native": {"panels": {"p": native}},
    }}
    write_tables(result, tmp_path)
    with (tmp_path / "scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["accuracy"] == ""
    assert rows[1]["accuracy"] == "0.75"
    assert rows[1]["metric"] == "JevBench native public accuracy"


def test_table_labels_published_baselines_and_does_not_invent_calibration(tmp_path):
    import csv

    record = jevbench_record(task())
    record["_eval_key"] = "a"
    incomplete, _ = score_panel([record], {})
    result = {
        "panels": [{"id": "jevbench/easy"}],
        "models": {"local": {"panels": {"jevbench/easy": incomplete}}},
        "official_jev_jevbench": {
            "provenance": "published_by_JevBench_not_rerun", "source_path": "source.json",
            "panels": {"jevbench/easy": {"questions": 48, "accuracy": 1.0}},
        },
    }
    write_tables(result, tmp_path)
    with (tmp_path / "scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["measurement"] == "published_by_JevBench_not_rerun"
    assert rows[1]["answered_brier"] == rows[1]["model_latency_median_ms"] == ""
    with (tmp_path / "accuracy.csv").open() as stream:
        matrix = list(csv.reader(stream))
    assert matrix[-1] == ["jev-1.13.0-published-by-jevbench", "1.0"]


def test_published_rejection_uses_the_same_accuracy_denominator(tmp_path):
    import csv

    records = [jevbench_record({**task(), "id": ident}) for ident in ("a", "b")]
    for record in records:
        record["_eval_key"] = record["_meta"]["id"]
    incomplete, _ = score_panel(records, {})
    result = {
        "panels": [{"id": "p"}], "models": {"local": {"panels": {"p": incomplete}}},
        "official_jev": [{
            "report": {"clean": {"n": 1, "acc": 1.0},
                       "coverage": {"requested_records": 2, "evaluated_records": 1, "rejected_records": 1}},
            "matching_panels": ["p"], "provenance": "published_by_Kev_not_rerun",
            "source_path": "reference.json",
        }],
    }
    write_tables(result, tmp_path)
    with (tmp_path / "scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["accuracy"] == "0.5"
