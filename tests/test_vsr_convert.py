"""Offline VSR conversion tests using original synthetic annotations and images."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from jevany.data import api_request, materialize
from jevany.suite import digest, load_split, read_json, read_manifest, validate_suite, write_json, write_jsonl
from scripts.convert_vsr import convert_jsonl, select_records


REVISION = "synthetic-revision"
LICENSES = [
    {"id": 4, "name": "Attribution License", "url": "http://creativecommons.org/licenses/by/2.0/"},
    {"id": 1, "name": "Attribution-NonCommercial-ShareAlike License",
     "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"},
]


def export_row(identifier=1, label=1, **extra):
    name = f"{identifier:012d}.jpg"
    return {"image": name, "image_link": f"http://images.cocodataset.org/train2017/{name}",
            "caption": "A square is left of a circle.", "label": label, "relation": "left of", **extra}


def inputs(tmp_path, rows=None):
    rows = rows if rows is not None else [export_row(1, 1), export_row(2, 0)]
    path, coco, images = tmp_path / "original.jsonl", tmp_path / "coco.json", tmp_path / "images"
    write_jsonl(path, rows)
    entries = {}
    for row in rows:
        identifier = int(row["image"].removesuffix(".jpg"))
        entries[row["image"]] = {
            "id": identifier, "file_name": row["image"], "coco_url": row["image_link"], "license": 4,
            "flickr_url": f"https://www.flickr.com/photos/synthetic/{identifier}",
        }
        split = row["image_link"].split("/")[-2]
        image = images / split / row["image"]
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), (identifier * 50 % 255, 30, 80)).save(image, format="JPEG")
    write_json(coco, {"images": list(entries.values()), "licenses": LICENSES})
    return path, coco, images


def convert(tmp_path, path, coco, images, **kwargs):
    return convert_jsonl(path, tmp_path / "suite", coco_annotations=[coco], image_root=images,
                         revision=REVISION, config=kwargs.pop("config", "random"),
                         split=kwargs.pop("split", "dev"), **kwargs)


@pytest.mark.parametrize("config", ["random", "zeroshot"])
@pytest.mark.parametrize("split", ["dev", "test"])
def test_preserves_annotation_and_media_with_locked_partitions(tmp_path, config, split):
    rows = [export_row(1, 1, caption="  A square is left of a circle.  ", reference_frame=None),
            export_row(2, 0, subj="square", obj="circle", vote_true_validator_id="[]")]
    path, coco, images = inputs(tmp_path, rows)
    output = convert(tmp_path, path, coco, images, config=config, split=split)
    partition = "development" if split == "dev" else "test"
    records = load_split(output, partition, allow_test=split == "test")
    manifest = read_manifest(output)
    assert validate_suite(output, allow_test=split == "test")["partitions"][partition]["records"] == 2
    assert manifest["trainable_sources"] == []
    assert manifest["eval_only_sources"] == ["vsr"]
    assert manifest["sources"]["vsr"]["input_sha256"] == digest(path)
    assert manifest["sources"]["vsr"]["coco_annotations"][0]["sha256"] == digest(coco)
    for original, record in zip(rows, records):
        question = record["questions"]["statement"]
        assert question["instructions"] == original["caption"]
        assert question["label"] is (original["label"] == 1)
        assert materialize(record)["questions"][0]["label"] == original["label"]
        request = api_request(record)
        assert set(request) == {"state", "questions", "media"}
        assert set(request["questions"]["statement"]) == {"type", "instructions"}
        assert isinstance(request["state"], str)
        metadata = record["_meta"]
        assert metadata["revision"] == REVISION
        assert metadata["config"] == config
        assert metadata["upstream_split"] == split
        assert metadata["relation"] == original["relation"]
        assert metadata["image_license"]["id"] == 4
        copied = Path(record["media"][0]["uri"])
        original_image = images / "train2017" / original["image"]
        assert copied.read_bytes() == original_image.read_bytes()
        assert digest(copied) == metadata["media_sha256"][0]
    if split == "test":
        with pytest.raises(ValueError, match="allow-test"):
            load_split(output, "test")
        with pytest.raises(ValueError, match="allow-test"):
            validate_suite(output)


def test_filter_is_resolved_url_not_name_or_id_and_excluded_images_not_needed(tmp_path):
    path, coco, images = inputs(tmp_path)
    metadata = read_json(coco)
    metadata["images"][1]["license"] = 1
    metadata["licenses"][0]["id"] = 99
    metadata["licenses"][0]["name"] = "Synthetic display name"
    metadata["images"][0]["license"] = 99
    write_json(coco, metadata)
    (images / "train2017" / export_row(2)["image"]).unlink()
    output = convert(tmp_path, path, coco, images)
    report = read_manifest(output)["selection"]
    assert report["input_records"] == 2
    assert report["retained_records"] == 1
    assert report["excluded_records"] == 1
    assert report["before"]["labels"] == {"0": 1, "1": 1}
    assert report["after"]["labels"] == {"1": 1}
    assert report["exclusions"][0]["input_line"] == 2
    assert report["exclusions"][0]["image_license"]["id"] == 1


@pytest.mark.parametrize("url", ["http://creativecommons.org/licenses/by/2.0/",
                                  "https://creativecommons.org/licenses/by/2.0/"])
def test_equivalent_license_urls(tmp_path, url):
    path, coco, images = inputs(tmp_path)
    metadata = read_json(coco)
    metadata["licenses"][0]["url"] = url
    write_json(coco, metadata)
    assert len(load_split(convert(tmp_path, path, coco, images), "development")) == 2


@pytest.mark.parametrize("url", ["https://creativecommons.org/licenses/by-nc/2.0/",
                                  "https://creativecommons.org/licenses/by-sa/2.0/",
                                  "https://creativecommons.org/licenses/by-nd/2.0/",
                                  "https://creativecommons.org.evil.test/licenses/by/2.0/",
                                  "https://creativecommons.org/licenses/by/2.0/?different=1"])
def test_misleading_id_and_name_do_not_authorize_images(tmp_path, url):
    path, coco, images = inputs(tmp_path)
    metadata = read_json(coco)
    metadata["licenses"][0]["url"] = url
    write_json(coco, metadata)
    with pytest.raises(ValueError, match="no records"):
        convert(tmp_path, path, coco, images)
    assert not (tmp_path / "suite").exists()


@pytest.mark.parametrize("label", [True, False, "0", "1", "False", 1.0, None, -1, 2, [], {}])
def test_rejects_non_integer_binary_labels_before_output(tmp_path, label):
    path, coco, images = inputs(tmp_path)
    write_jsonl(path, [export_row(1), export_row(2, label)])
    with pytest.raises(ValueError, match=r"original.jsonl:2:.*label"):
        convert(tmp_path, path, coco, images)
    assert not (tmp_path / "suite").exists()


@pytest.mark.parametrize("field,value", [("caption", ""), ("relation", None), ("image", "../escape.jpg"),
                                         ("image_link", "http://evil.test/train2017/000000000001.jpg"),
                                         ("image_link", "http://images.cocodataset.org/val2017/000000000001.jpg")])
def test_rejects_invalid_or_mismatched_vsr_rows(tmp_path, field, value):
    path, coco, images = inputs(tmp_path)
    row = export_row(1)
    row[field] = value
    write_jsonl(path, [row])
    with pytest.raises(ValueError):
        convert(tmp_path, path, coco, images)
    assert not (tmp_path / "suite").exists()


@pytest.mark.parametrize("case", ["missing_image", "unknown_license", "conflicting_license", "conflicting_image",
                                  "mismatched_id", "invalid_license_url", "missing_flickr"])
def test_rejects_bad_coco_metadata(tmp_path, case):
    path, coco, images = inputs(tmp_path)
    metadata = read_json(coco)
    if case == "missing_image":
        metadata["images"] = []
    elif case == "unknown_license":
        metadata["images"][0]["license"] = 400
    elif case == "conflicting_license":
        metadata["licenses"].append({**metadata["licenses"][0], "name": "Conflicting"})
    elif case == "conflicting_image":
        metadata["images"].append({**metadata["images"][0], "license": 1})
    elif case == "mismatched_id":
        metadata["images"][0]["id"] = 100
    elif case == "invalid_license_url":
        metadata["licenses"][0]["url"] = "not-a-url"
    else:
        metadata["images"][0].pop("flickr_url")
    write_json(coco, metadata)
    with pytest.raises(ValueError):
        convert(tmp_path, path, coco, images)
    assert not (tmp_path / "suite").exists()


def test_consistent_metadata_files_and_val_images(tmp_path):
    name = "000000000001.jpg"
    path, coco, images = inputs(tmp_path, [export_row(1, image_link=f"http://images.cocodataset.org/val2017/{name}")])
    second = tmp_path / "second.json"
    write_json(second, read_json(coco))
    output = convert_jsonl(path, tmp_path / "suite", coco_annotations=[coco, second], image_root=images,
                           revision=REVISION, config="random", split="dev")
    assert len(load_split(output, "development")) == 1
    assert len(read_manifest(output)["sources"]["vsr"]["coco_annotations"]) == 2


def test_repeated_captions_and_same_image_remain_distinct_and_grouped(tmp_path):
    path, coco, images = inputs(tmp_path, [export_row(1), export_row(1, 0), export_row(1)])
    first = load_split(convert(tmp_path, path, coco, images), "development")
    assert len({row["_meta"]["id"] for row in first}) == 3
    assert len({row["_meta"]["group_id"] for row in first}) == 1
    assert len(list((tmp_path / "suite" / "media").iterdir())) == 1
    other = convert_jsonl(path, tmp_path / "other", coco_annotations=[coco], image_root=images,
                          revision=REVISION, config="zeroshot", split="test")
    second = load_split(other, "test", allow_test=True)
    assert first[0]["_meta"]["group_id"] == second[0]["_meta"]["group_id"]


@pytest.mark.parametrize("case", ["missing", "corrupt", "symlink_escape"])
def test_missing_corrupt_or_escaping_selected_images_do_not_create_output(tmp_path, case):
    path, coco, images = inputs(tmp_path)
    image = images / "train2017" / export_row(1)["image"]
    if case == "corrupt":
        image.write_bytes(b"not an image")
    else:
        image.unlink()
        if case == "symlink_escape":
            outside = tmp_path / "outside.jpg"
            Image.new("RGB", (12, 12)).save(outside)
            image.symlink_to(outside)
    with pytest.raises((OSError, ValueError)):
        convert(tmp_path, path, coco, images)
    assert not (tmp_path / "suite").exists()


@pytest.mark.parametrize("kind", ["directory", "file", "broken_symlink"])
def test_refuses_existing_outputs(tmp_path, kind):
    path, coco, images = inputs(tmp_path)
    target = tmp_path / "suite"
    if kind == "directory":
        target.mkdir()
    elif kind == "file":
        target.write_text("keep me", encoding="utf-8")
    else:
        target.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        convert(tmp_path, path, coco, images)
    assert target.exists() or target.is_symlink()


def test_determinism_and_media_control_compatibility(tmp_path):
    from scripts.eval_multimodal import ablate, verify_media, verify_suite_artifacts

    path, coco, images = inputs(tmp_path)
    output = convert(tmp_path, path, coco, images)
    other = convert_jsonl(path, tmp_path / "other", coco_annotations=[coco], image_root=images,
                          revision=REVISION, config="random", split="dev")
    assert (output / "development.jsonl").read_bytes() == (other / "development.jsonl").read_bytes()
    assert (output / "manifest.json").read_bytes() == (other / "manifest.json").read_bytes()
    rows = load_split(output, "development")
    assert verify_media(rows) == 2
    assert len(verify_suite_artifacts(output)) == 2
    shuffled = ablate(rows, "shuffled")
    assert all(left["media"] != right["media"] for left, right in zip(rows, shuffled))
    assert rows == ablate(rows, "full")
    blank = ablate(rows, "blank", blank_image=images / "train2017" / export_row(1)["image"])
    assert all(row["media"] == blank[0]["media"] for row in blank)
    Path(rows[0]["media"][0]["uri"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_media(rows)


def test_empty_input_and_empty_metadata_are_rejected(tmp_path):
    path, coco, images = inputs(tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        select_records(path, [])
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        convert(tmp_path, path, coco, images)
    assert not (tmp_path / "suite").exists()


@pytest.mark.parametrize("config,split,revision", [("other", "dev", REVISION), ("random", "train", REVISION),
                                                  ("random", "dev", "")])
def test_rejects_invalid_conversion_arguments(tmp_path, config, split, revision):
    path, coco, images = inputs(tmp_path)
    with pytest.raises(ValueError):
        convert_jsonl(path, tmp_path / "suite", coco_annotations=[coco], image_root=images,
                      revision=revision, config=config, split=split)
    assert not (tmp_path / "suite").exists()


def test_cli_conversion_and_help_are_offline(tmp_path):
    path, coco, images = inputs(tmp_path)
    guard = """
import sys

def no_network(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise AssertionError('network access is forbidden')

sys.addaudithook(no_network)
from scripts.convert_vsr import main
main()
assert 'torch' not in sys.modules
"""
    arguments = [str(path), "--coco-annotations", str(coco), "--image-root", str(images),
                 "--out", str(tmp_path / "cli"), "--revision", REVISION, "--config", "random", "--split", "dev"]
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", guard, *arguments], cwd=repo,
                            capture_output=True, text=True, check=True)
    assert str(tmp_path / "cli") in result.stdout
    assert len(load_split(tmp_path / "cli", "development")) == 2
    help_result = subprocess.run([sys.executable, "-c", guard, "--help"], cwd=repo,
                                 capture_output=True, text=True, check=True)
    assert "--coco-annotations" in help_result.stdout
