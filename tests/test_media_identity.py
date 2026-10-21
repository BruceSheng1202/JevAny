"""Media identity is shared by dataset deduplication and overlap checks."""
import copy

import pytest

from jevany.data import load_records
from jevany.suite import digest, load_split, semantic_hash, validate_suite, write_json, write_jsonl


TEXT_HASH = "15504894a79f9615b2dfc9f0a2b0fb3db86d11f24ea0c9cae874bf781bb11abb"


def record(identifier, media=()):
    return {
        "state": "Inspect the evidence.",
        "questions": {"q": {"type": "choice", "instructions": "Which object?",
                            "criteria": {"a": "A", "b": "B"}, "label": "a"}},
        "media": list(media),
        "_meta": {"source": "custom", "id": identifier, "group_id": identifier,
                  "text_sha256": TEXT_HASH},
    }


def make_suite(tmp_path, partitions):
    root = tmp_path / "suite"
    root.mkdir()
    files = {}
    for split, rows in partitions.items():
        path = root / f"{split}.jsonl"
        write_jsonl(path, rows)
        files[path.name] = {"sha256": digest(path), "records": len(rows)}
    write_json(root / "manifest.json", {"files": files, "trainable_sources": ["custom"]})
    return root


def test_text_hash_preserves_legacy_digest_and_normalization():
    row = record("first")
    assert semantic_hash(row) == TEXT_HASH
    del row["media"]
    row["state"] = "inspect  THE evidence."
    row["questions"]["q"].update(label="b", target={"b": 1.0}, src="other")
    row["_meta"] = {"text_sha256": "old provenance"}
    assert semantic_hash(row) == TEXT_HASH


@pytest.mark.parametrize("first_media,second_media", [
    ([{"type": "image", "uri": "a.png"}], [{"type": "image", "uri": "b.png"}]),
    ([{"type": "video", "uri": "a.mp4"}], [{"type": "video", "uri": "b.mp4"}]),
    ([{"type": "image", "uri": "A.png"}], [{"type": "image", "uri": "a.png"}]),
    ([{"type": "image", "uri": "a  b.png"}], [{"type": "image", "uri": "a b.png"}]),
    ([{"type": "image", "uri": "a.png"}], [{"type": "video", "uri": "a.png"}]),
    ([{"type": "image", "uri": "a.png"}, {"type": "video", "uri": "b.mp4"}],
     [{"type": "video", "uri": "b.mp4"}, {"type": "image", "uri": "a.png"}]),
    ([{"type": "image", "uri": "a.png"}],
     [{"type": "image", "uri": "a.png"}, {"type": "image", "uri": "a.png"}]),
    ([], [{"type": "image", "uri": "a.png"}]),
], ids=["images", "videos", "uri-case", "uri-whitespace", "type", "order", "multiplicity", "text-vs-media"])
def test_builder_deduplication_and_exclusions_use_media(first_media, second_media):
    from jevany.datasets.build_sft import unique

    first = record("first", first_media)
    second = record("second", second_media)
    duplicate = copy.deepcopy(first)
    duplicate["state"] = "inspect  THE evidence."
    duplicate["questions"]["q"].update(label="b", target={"b": 1.0}, src="other")
    duplicate["_meta"] = {"id": "duplicate", "text_sha256": "different provenance",
                          "media_sha256": ["different provenance"]}
    duplicate["media"] = [dict(reversed(list(item.items()))) for item in duplicate["media"]]

    kept = unique([first, second, duplicate], set())
    assert [row["_meta"]["id"] for row in kept] == ["first", "second"]
    assert first["_meta"]["text_sha256"] != second["_meta"]["text_sha256"]
    assert unique([duplicate, second], {semantic_hash(first)}) == [second]


@pytest.mark.parametrize("overlap", ["none", "media", "group"])
def test_suite_overlap_uses_media_and_preserves_group_checks(tmp_path, overlap):
    train = record("train", [{"type": "image", "uri": "a.png"}])
    development = record("development", [{"type": "image", "uri": "a.png" if overlap == "media" else "b.png"}])
    development["questions"]["q"]["label"] = "b"
    if overlap == "group":
        development["_meta"]["group_id"] = "train"
    root = make_suite(tmp_path, {"train": [train], "development": [development]})
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    if overlap == "none":
        assert set(validate_suite(root)["partitions"]) == {"train", "development"}
    else:
        kind = "decision text" if overlap == "media" else "group"
        with pytest.raises(ValueError, match=f"{kind} overlap between train and development"):
            validate_suite(root)
    assert load_split(root, "train")[0]["_meta"]["text_sha256"] == TEXT_HASH
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


@pytest.mark.parametrize("training_source", ["data", "suite", "replay"])
@pytest.mark.parametrize("same_media", [False, True])
def test_training_preflight_compares_resolved_media(tmp_path, monkeypatch, training_source, same_media):
    from jevany.train import main

    train = record("train", [{"type": "image", "uri": "a.png"}])
    root = make_suite(tmp_path, {"train": [train]})
    # Preflight checks file existence; media decoding happens when training starts.
    (root / "a.png").write_bytes(b"fixture")
    (root / "b.png").write_bytes(b"fixture")
    evaluation = record("evaluation", [{"type": "image", "uri": str(root / ("a.png" if same_media else "b.png"))}])
    evaluation["questions"]["q"]["label"] = "b"
    eval_data = tmp_path / "evaluation.jsonl"
    write_jsonl(eval_data, [evaluation])
    out = tmp_path / "run"
    args = ["--eval-data", str(eval_data), "--eval-every-steps", "1",
            "--out", str(out), "--multimodal", "--dry-run"]
    if training_source == "data":
        args += ["--data", str(root / "train.jsonl")]
    else:
        args += ["--suite", str(root), "--base-revision", "fixture"]
        if training_source == "replay":
            extra = record("extra")
            extra["state"] = "Unrelated text."
            data = tmp_path / "extra.jsonl"
            write_jsonl(data, [extra])
            args += ["--data", str(data), "--replay", "1"]
    monkeypatch.setattr("jevany.train.load_preprocessor",
                        lambda *args, **kwargs: pytest.fail("dry-run must not load a model"))

    if same_media:
        with pytest.raises(ValueError, match="overlaps training data"):
            main(args)
    else:
        assert main(args) == out
    assert not out.exists()


@pytest.mark.parametrize("media", [
    [{"type": "audio", "uri": "a.png"}],
    [{"type": "image", "uri": ""}],
    [{"type": "image"}],
])
def test_invalid_media_still_fails_record_validation(tmp_path, media):
    path = tmp_path / "invalid.jsonl"
    write_jsonl(path, [record("invalid", media)])
    with pytest.raises(ValueError, match=r"invalid\.jsonl:1:") as error:
        load_records(path)
    assert "media" in str(error.value)
