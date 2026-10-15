"""Attribution conversion preserves bundle labels and needs no models or network."""
import copy
import json
import subprocess
import sys

import pytest

from jevany.data import api_request, materialize
from jevany.suite import (
    digest, load_split, read_manifest, validate_suite, validate_training, write_json, write_jsonl,
)
from scripts.convert_attributionbench import SOURCE, convert_jsonl, main


def export_row(**changes):
    return {
        "id": "example-1", "src_dataset": "synthetic",
        "question": "When is the shop open?",
        "claim": "The shop is open on Monday.",
        "references": ["The shop is open on Monday."],
        "attribution_label": "attributable",
        "response": "A response that is not supplied as evidence.",
        "claim_raw_string": "An unused raw claim.",
        "citation_links": ["A link that is not supplied as evidence."],
        **changes,
    }


def convert_rows(tmp_path, rows, *, split="dev", name="suite"):
    export = tmp_path / "export.jsonl"
    write_jsonl(export, rows)
    output = convert_jsonl(export, tmp_path / name, revision="fixture-r1", split=split)
    partition = "development" if split == "dev" else "test"
    return export, output, load_split(output, partition, allow_test=partition == "test")


@pytest.mark.parametrize("split,partition", [("dev", "development"), ("test", "test"), ("test_ood", "test")])
def test_conversion_preserves_labels_bundles_and_provenance(tmp_path, split, partition):
    original = [
        export_row(claim="  The caf\u00e9 is open.\n", references=[" First\nreference. ", "Second", "Second"]),
        export_row(id="example-2", attribution_label="not attributable"),
    ]
    before = copy.deepcopy(original)
    export, output, rows = convert_rows(tmp_path, original, split=split)
    assert original == before
    assert [row["questions"]["support"]["label"] for row in rows] == [True, False]
    for input_row, row in zip(original, rows):
        assert row["state"] == {"claim": input_row["claim"], "references": input_row["references"]}
        assert row["questions"]["support"]["src"] == "attributionbench_synthetic"
        assert row["_meta"]["id"] == f"{SOURCE}/{input_row['id']}"
        assert row["_meta"]["original_id"] == input_row["id"]
        assert row["_meta"]["src_dataset"] == "synthetic"
        assert row["_meta"]["source"] == SOURCE
        assert row["_meta"]["variant"] == "clean"
        assert row["_meta"]["split"] == partition
        assert row["_meta"]["upstream_split"] == split
        assert row["_meta"]["revision"] == "fixture-r1"
        request = api_request(row)
        assert set(request) == {"state", "questions"}
        assert set(request["questions"]["support"]) == {"type", "instructions"}
        assert materialize(row)["questions"][0]["label"] == int(input_row["attribution_label"] == "attributable")
    assert [row["_meta"]["input_line"] for row in rows] == [1, 2]
    assert rows[0]["_meta"]["group_id"] == rows[1]["_meta"]["group_id"]
    manifest = read_manifest(output)
    source = manifest["sources"][SOURCE]
    assert source["input_sha256"] == digest(export)
    assert source["repository"] == "osunlp/AttributionBench"
    assert source["upstream_split"] == split
    assert manifest["trainable_sources"] == []
    assert manifest["eval_only_sources"] == [SOURCE]
    assert manifest["files"][f"{partition}.jsonl"] == {
        "records": 2, "questions": 2, "sha256": digest(output / f"{partition}.jsonl"),
    }
    assert validate_suite(output, allow_test=partition == "test")["partitions"][partition]["records"] == 2
    assert sorted(path.name for path in output.iterdir()) == sorted([f"{partition}.jsonl", "manifest.json"])
    with pytest.raises(ValueError, match="eval-only"):
        validate_training(rows, manifest)


@pytest.mark.parametrize("label", ["attributable", "not attributable"])
def test_empty_reference_bundle_keeps_original_label(tmp_path, label):
    _, _, rows = convert_rows(tmp_path, [export_row(references=[], attribution_label=label)])
    assert rows[0]["state"]["references"] == []
    assert rows[0]["questions"]["support"]["label"] is (label == "attributable")


def test_grouping_is_independent_of_ids_claims_and_output_split(tmp_path):
    rows = [export_row(), export_row(id="example-2", claim="The shop is closed.")]
    _, _, development = convert_rows(tmp_path, rows)
    _, _, test = convert_rows(tmp_path, rows, split="test", name="test-suite")
    groups = {row["_meta"]["group_id"] for row in development + test}
    assert len(groups) == 1


@pytest.mark.parametrize("question", [None, "", "  "])
def test_reference_group_fallback_is_source_scoped(tmp_path, question):
    rows = [export_row(id=str(index), question=question, **changes) for index, changes in enumerate([
        {}, {"claim": "A different claim."}, {"src_dataset": "other"}, {"references": ["Other evidence."]},
    ])]
    _, _, converted = convert_rows(tmp_path, rows)
    groups = [row["_meta"]["group_id"] for row in converted]
    assert groups[0] == groups[1]
    assert len(set(groups)) == 3


@pytest.mark.parametrize("field", ["id", "src_dataset", "claim", "references", "attribution_label"])
def test_missing_fields_fail_before_output_creation(tmp_path, field):
    row = export_row()
    del row[field]
    export = tmp_path / "export.jsonl"
    write_jsonl(export, [row])
    output = tmp_path / "suite"
    with pytest.raises(ValueError, match="export.jsonl:1:"):
        convert_jsonl(export, output, revision="r1", split="dev")
    assert not output.exists()


@pytest.mark.parametrize("changes", [
    {"id": " "}, {"id": 1}, {"src_dataset": None}, {"claim": ""}, {"claim": []},
    {"references": None}, {"references": "Evidence"}, {"references": [1]},
    {"attribution_label": True}, {"attribution_label": "supported"}, {"question": []},
])
def test_bad_rows_report_physical_lines_without_partial_output(tmp_path, changes):
    export = tmp_path / "export.jsonl"
    export.write_text("\n" + json.dumps(export_row()) + "\n\n" + json.dumps(export_row(**changes)) + "\n")
    output = tmp_path / "suite"
    with pytest.raises(ValueError, match="export.jsonl:4:"):
        convert_jsonl(export, output, revision="r1", split="dev")
    assert not output.exists()


@pytest.mark.parametrize("content", ["", "\n\n", "[]\n", "null\n", '{"claim":'])
def test_empty_nonobject_or_invalid_json_is_rejected(tmp_path, content):
    export = tmp_path / "export.jsonl"
    export.write_text(content)
    output = tmp_path / "suite"
    with pytest.raises(ValueError, match="export.jsonl"):
        convert_jsonl(export, output, revision="r1", split="dev")
    assert not output.exists()


def test_duplicate_ids_are_rejected_before_output_creation(tmp_path):
    export = tmp_path / "export.jsonl"
    write_jsonl(export, [export_row(), export_row(claim="Another claim.")])
    with pytest.raises(ValueError, match="export.jsonl:2:.*duplicate original id"):
        convert_jsonl(export, tmp_path / "suite", revision="r1", split="dev")
    assert not (tmp_path / "suite").exists()


@pytest.mark.parametrize("changes", [{"revision": " "}, {"revision": None}, {"split": "train"}, {"split": "development"}])
def test_invalid_arguments_do_not_read_input_or_create_output(tmp_path, changes):
    with pytest.raises(ValueError):
        convert_jsonl(tmp_path / "missing.jsonl", tmp_path / "suite", **{"revision": "r1", "split": "dev", **changes})
    assert not (tmp_path / "suite").exists()


def test_repeated_conversion_is_deterministic(tmp_path):
    export, first, _ = convert_rows(tmp_path, [export_row()])
    second = convert_jsonl(export, tmp_path / "second", revision="fixture-r1", split="dev")
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_cli_refuses_overwrite_and_reports_invalid_input(tmp_path, capsys):
    export, output, _ = convert_rows(tmp_path, [export_row()])
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    args = [str(export), "--out", str(output), "--revision", "r1", "--split", "dev"]
    with pytest.raises(SystemExit) as caught:
        main(args)
    assert caught.value.code == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    fresh = tmp_path / "fresh"
    export.write_text("not json\n")
    with pytest.raises(SystemExit) as caught:
        main([str(export), "--out", str(fresh), "--revision", "r1", "--split", "dev"])
    assert caught.value.code == 2 and "export.jsonl:1:" in capsys.readouterr().err
    assert not fresh.exists()


def test_generated_test_suite_is_locked(tmp_path):
    _, output, _ = convert_rows(tmp_path, [export_row()], split="test")
    with pytest.raises(ValueError, match="allow-test"):
        load_split(output, "test")
    with pytest.raises(ValueError, match="allow-test"):
        validate_suite(output)


def test_shared_question_is_detected_when_partitions_are_checked_together(tmp_path):
    _, development, _ = convert_rows(tmp_path, [export_row()])
    _, test, _ = convert_rows(tmp_path, [export_row(id="other", claim="A different claim.")], split="test", name="test-suite")
    (development / "test.jsonl").write_bytes((test / "test.jsonl").read_bytes())
    manifest = read_manifest(development)
    manifest["files"].update(read_manifest(test)["files"])
    write_json(development / "manifest.json", manifest)
    with pytest.raises(ValueError, match="group overlap"):
        validate_suite(development, allow_test=True)


def test_prediction_rows_accept_converted_metadata(tmp_path):
    from jevany.benchmark import prediction_rows

    _, _, rows = convert_rows(tmp_path, [export_row()])
    scored = prediction_rows(rows[0], {"probabilities": {"support": {"false": 0.1, "true": 0.9}}})
    assert scored[0]["label"] == 1
    assert scored[0]["variant"] == "clean"
    assert scored[0]["task"] == "attributionbench_synthetic"


def test_cli_runs_without_optional_imports_or_network(tmp_path):
    export = tmp_path / "export.jsonl"
    write_jsonl(export, [export_row()])
    script = """
import sys
class NoOptionalImports:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "transformers", "datasets", "boto3"}:
            raise ImportError("optional dependency unavailable: " + name)
        return None
sys.meta_path.insert(0, NoOptionalImports())
def no_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("conversion must not access the network")
sys.addaudithook(no_network)
from scripts.convert_attributionbench import main
main([sys.argv[1], "--out", sys.argv[2], "--revision", "r1", "--split", "dev"])
"""
    output = tmp_path / "suite"
    result = subprocess.run([sys.executable, "-c", script, str(export), str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(output)
    assert len(load_split(output, "development")) == 1
