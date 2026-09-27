"""Protocol and recovery checks without model weights, accounts, or network."""
import json

import pytest

from jevany.external_eval import load_completed, run_requests
from scripts.build_external_eval import PUBLIC_CONTEXT, jevbench_record, request_key, validate_file
from scripts.report_external_eval import score_panel, typesafe_metrics


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


def test_typesafe_weights_cases_equally_and_penalizes_missing():
    records, rows = [], []
    for ident, group in (("a", "case1"), ("b", "case1"), ("c", "case2")):
        record = jevbench_record({**task(), "id": ident, "group": group})
        record["_meta"]["target"] = {"false": 0.0, "true": 1.0}
        records.append(record)
        if ident != "c":
            rows.append({"id": ident, "keys": ["false", "true"], "p": [0.0, 1.0]})
    result = typesafe_metrics(records, rows)
    assert result["evaluated"]["equal_case_modal_agreement"] == 1
    assert result["all_rows"]["equal_case_modal_agreement"] == .5
    assert result["all_rows"]["equal_case_total_variation"] == .5
