"""Media identity is shared by dataset deduplication and overlap checks."""
import copy
import json
from pathlib import Path

import pytest

from jevany.data import load_records
from jevany.suite import digest, load_split, read_jsonl, read_manifest, semantic_hash, validate_suite, write_json, write_jsonl


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


@pytest.mark.parametrize("media_type", ["image", "video", None])
def test_rlcr_can_exclude_its_own_parent_partition(tmp_path, media_type):
    from jevany.datasets import build_rlcr

    media = [{"type": media_type, "uri": "asset"}] if media_type else []
    root = make_suite(tmp_path, {"train": [record("train", media)]})
    (root / "asset").write_bytes(b"fixture")
    output = tmp_path / "rlcr"
    quotas = {**dict.fromkeys(build_rlcr.QUOTAS, 0), "core": 1}
    with pytest.raises(ValueError, match="core: requested 1 records but only 0 are available"):
        build_rlcr.main(["--suite", str(root), "--out", str(output),
                         "--exclude", str(root / "train.jsonl"), "--quotas", json.dumps(quotas)])
    assert not output.exists()


@pytest.fixture
def offline_sft(monkeypatch):
    from jevany.datasets import build_sft

    monkeypatch.setattr(build_sft, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(build_sft, "admitted", lambda *args: True)
    monkeypatch.setattr(build_sft, "load_source",
                        lambda *args: pytest.fail("unexpected dataset load"))
    return build_sft


def sft_args(output):
    return ["--out", str(output)] + [
        arg for flag in ("helpsteer", "agent", "hard-text", "scienceqa", "aokvqa", "video", "eval-per-source")
        for arg in (f"--{flag}", "0")
    ]


def test_sft_included_files_keep_their_own_relative_media(tmp_path, offline_sft):
    output = tmp_path / "output"
    args = sft_args(output)
    expected = set()
    for name in ("one", "two"):
        directory = tmp_path / name
        directory.mkdir()
        asset = directory / "asset.png"
        asset.write_bytes(b"fixture")
        path = directory / "train.jsonl"
        write_jsonl(path, [record(name, [{"type": "image", "uri": "asset.png"}])])
        args += ["--include-train", str(path)]
        expected.add(str(asset))

    offline_sft.main(args)

    rows = load_split(output, "train")
    assert {row["media"][0]["uri"] for row in rows} == expected
    assert all(not Path(row["media"][0]["uri"]).is_absolute()
               for row in read_jsonl(output / "train.jsonl"))


@pytest.mark.parametrize("same_file", [False, True])
def test_sft_exclusions_resolve_media_against_the_exclusion_file(tmp_path, offline_sft, same_file):
    included = tmp_path / "included"
    excluded = tmp_path / "excluded"
    included.mkdir()
    excluded.mkdir()
    for directory in (included, excluded):
        (directory / "asset.png").write_bytes(b"fixture")
    train = record("train", [{"type": "image", "uri": "asset.png"}])
    write_jsonl(included / "train.jsonl", [train])
    exclusion = copy.deepcopy(train)
    exclusion["media"][0]["uri"] = "../included/asset.png" if same_file else "asset.png"
    del exclusion["questions"]["q"]["label"]
    del exclusion["_meta"]
    write_jsonl(excluded / "exclude.jsonl", [exclusion])
    output = tmp_path / "output"

    offline_sft.main(sft_args(output) + ["--include-train", str(included / "train.jsonl"),
                                       "--exclude", str(excluded / "exclude.jsonl")])

    assert len(read_jsonl(output / "train.jsonl")) == (0 if same_file else 1)


@pytest.mark.parametrize("extra_input", ["none", "exclude", "include"])
def test_sft_generated_media_uses_the_output_directory(tmp_path, monkeypatch, offline_sft, extra_input):
    from PIL import Image

    source_row = {"image": Image.new("RGB", (1, 1)), "question": "Which object?",
                  "choices": ["A", "B"], "answer": 0}
    output = tmp_path / "output"
    generated = offline_sft.visual_record(source_row, "scienceqa", 0, "train", output)
    monkeypatch.setattr(offline_sft, "load_source",
                        lambda source, split: ([source_row] * (1 if split == "train" else 2)
                                                if source == "scienceqa" else []))
    args = sft_args(output) + ["--scienceqa", "1", "--eval-per-source", "1", "--resume"]
    if extra_input == "exclude":
        excluded = copy.deepcopy(generated)
        excluded["media"][0]["uri"] = str(Path("output") / excluded["media"][0]["uri"])
        path = tmp_path / "exclude.jsonl"
        write_jsonl(path, [excluded])
        args += ["--exclude", str(path)]
    elif extra_input == "include":
        included = tmp_path / "included"
        imported = offline_sft.visual_record(source_row, "scienceqa", 0, "train", included)
        path = included / "train.jsonl"
        write_jsonl(path, [imported])
        args += ["--include-train", str(path)]

    offline_sft.main(args)

    counts = {"train": {"none": 1, "exclude": 0, "include": 2}[extra_input],
              "calibration": 1, "development": 1}
    for split, count in counts.items():
        raw = read_jsonl(output / f"{split}.jsonl")
        assert len(raw) == count
        assert all(not Path(item["uri"]).is_absolute() for row in raw for item in row["media"])
        assert all(Path(item["uri"]).is_file() for row in load_split(output, split) for item in row["media"])


def test_sft_generated_partitions_are_reproducible_across_output_roots(tmp_path, monkeypatch, offline_sft):
    from PIL import Image

    source_row = {"image": Image.new("RGB", (1, 1)), "question": "Which object?",
                  "choices": ["A", "B"], "answer": 0}
    monkeypatch.setattr(offline_sft, "load_source",
                        lambda source, split: ([source_row] * (1 if split == "train" else 2)
                                                if source == "scienceqa" else []))
    outputs = [tmp_path / name for name in ("first", "second")]
    for output in outputs:
        offline_sft.main(sft_args(output) + ["--scienceqa", "1", "--eval-per-source", "1"])

    for split in ("train", "calibration", "development"):
        filename = f"{split}.jsonl"
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
        for output in outputs:
            rows = read_jsonl(output / filename)
            assert len(rows) == 1
            assert rows[0]["_meta"]["text_sha256"] == semantic_hash(rows[0])
        assert read_manifest(outputs[0])["files"][filename] == read_manifest(outputs[1])["files"][filename]
