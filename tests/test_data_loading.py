"""Labelled JSONL keeps Unicode evidence and reports physical input lines."""
import json

import pytest

from jevany.data import load_records, validate_dataset


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_separators_inside_json_strings_are_preserved(tmp_path, separator):
    row = {
        "state": {"note": f"north{separator}south"},
        "questions": {
            "q": {
                "type": "choice",
                "instructions": f"Choose{separator}one",
                "criteria": {"a": f"left{separator}right", "b": "other"},
                "label": "a",
            }
        },
    }
    path = tmp_path / "records.jsonl"
    path.write_text("\n" + json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    records = load_records(path, source="fixture")

    assert len(records) == 1
    assert records[0]["state"] == row["state"]
    question = records[0]["questions"]["q"]
    assert question["instructions"] == row["questions"]["q"]["instructions"]
    assert question["criteria"] == row["questions"]["q"]["criteria"]
    assert records[0]["_meta"]["row"] == 1
    assert records[0]["_meta"]["id"] == "fixture/1"
    assert validate_dataset(path)["questions"] == 1


def test_invalid_json_after_unicode_evidence_reports_physical_line(tmp_path):
    row = {"state": "first\u2028second", "questions": {"q": {"type": "noul", "label": True}}}
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n" + json.dumps(row, ensure_ascii=False) + '\n\n{"state":',
        encoding="utf-8", newline="\r\n",
    )

    with pytest.raises(ValueError, match=r"records.jsonl:4:"):
        load_records(path)
