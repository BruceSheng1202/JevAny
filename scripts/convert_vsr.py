#!/usr/bin/env python3
"""Convert local VSR annotations into a COCO-metadata CC BY 2.0 evaluation subset."""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from jevany.data import materialize
from jevany.datasets.convert import record
from jevany.suite import digest, read_json, validate_suite, write_json, write_jsonl


SOURCE = "vsr"
REPOSITORY = "https://github.com/cambridgeltl/visual-spatial-reasoning"
SPLITS = {"dev": "development", "test": "test"}
CONFIGS = ("random", "zeroshot")
IMAGE_NAME = re.compile(r"[0-9]{12}\.jpg\Z")
STATE = "Judge whether the statement is true in the supplied image. Use only the image."


def web_url(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty HTTP(S) URL")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ValueError(f"{field} must be an absolute HTTP(S) URL without credentials")
    return parts


def image_location(name, url):
    if not isinstance(name, str) or not IMAGE_NAME.fullmatch(name):
        raise ValueError("image filename must be twelve digits followed by .jpg")
    parts = web_url(url, "COCO image URL")
    location = parts.path.removeprefix("/").split("/")
    if (parts.netloc != "images.cocodataset.org" or parts.query or parts.fragment
            or len(location) != 2 or location[0] not in ("train2017", "val2017")
            or location[1] != name):
        raise ValueError("COCO image URL must match the filename and train2017/val2017 layout")
    return Path(*location)


def is_cc_by_2_0(url):
    parts = web_url(url, "image license URL")
    return (parts.netloc == "creativecommons.org" and parts.path == "/licenses/by/2.0/"
            and not parts.query and not parts.fragment)


def load_coco_metadata(paths):
    """Merge consistent image/license tables; reject missing or conflicting mappings."""
    images, licenses = {}, {}
    for path in paths:
        data = read_json(path)
        if not isinstance(data, dict) or any(not isinstance(data.get(key), list) for key in ("images", "licenses")):
            raise ValueError(f"{path}: COCO metadata requires images and licenses lists")
        for entry in data["licenses"]:
            if (not isinstance(entry, dict) or type(entry.get("id")) is not int
                    or entry["id"] < 0 or not isinstance(entry.get("name"), str) or not entry["name"].strip()):
                raise ValueError(f"{path}: invalid COCO license record")
            web_url(entry.get("url"), "image license URL")
            value = {key: entry[key] for key in ("id", "name", "url")}
            previous = licenses.setdefault(entry["id"], value)
            if previous != value:
                raise ValueError(f"{path}: conflicting license ID {entry['id']}")
        for entry in data["images"]:
            if (not isinstance(entry, dict) or type(entry.get("id")) is not int or entry["id"] < 0
                    or type(entry.get("license")) is not int):
                raise ValueError(f"{path}: invalid COCO image record")
            name = entry.get("file_name")
            image_location(name, entry.get("coco_url"))
            if int(name.removesuffix(".jpg")) != entry["id"]:
                raise ValueError(f"{path}: COCO image ID does not match filename")
            value = {key: entry.get(key) for key in ("id", "file_name", "license", "coco_url", "flickr_url")}
            previous = images.setdefault(name, value)
            if previous != value:
                raise ValueError(f"{path}: conflicting image ID {entry['id']}")
    for image in images.values():
        if image["license"] not in licenses:
            raise ValueError(f"unknown license ID {image['license']} for COCO image {image['id']}")
    return images, licenses


def distribution(rows):
    return {
        "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "relations": dict(sorted(Counter(row["relation"] for row in rows).items())),
    }


def select_records(path, coco_annotations):
    """Validate and select annotations without reading photographs or writing outputs."""
    paths = [Path(value) for value in coco_annotations]
    if not paths:
        raise ValueError("at least one COCO annotation file is required")
    images, licenses = load_coco_metadata(paths)
    selected, originals, exclusions = [], [], []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each VSR row must be an object")
                for field in ("caption", "relation"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"{field} must be a non-empty string")
                if type(row.get("label")) is not int or row["label"] not in (0, 1):
                    raise ValueError("label must be the integer 0 or 1")
                location = image_location(row.get("image"), row.get("image_link"))
                image = images.get(row["image"])
                if image is None:
                    raise ValueError(f"missing COCO image metadata for {row['image']}")
                if location != image_location(image["file_name"], image["coco_url"]):
                    raise ValueError("VSR and COCO image locations disagree")
                license_record = licenses[image["license"]]
                if is_cc_by_2_0(license_record["url"]):
                    flickr = web_url(image.get("flickr_url"), "original Flickr URL")
                    if not any(flickr.hostname == host or flickr.hostname.endswith("." + host)
                               for host in ("flickr.com", "staticflickr.com")):
                        raise ValueError("original Flickr URL must identify a Flickr host")
                    selected.append({"row": row, "image": image, "license": license_record,
                                     "input_line": line_number, "location": location})
                else:
                    exclusions.append({"input_line": line_number, "coco_image_id": image["id"],
                                       "image_license": license_record, "reason": "not_cc_by_2_0"})
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            originals.append(row)
    if not originals:
        raise ValueError(f"{path}: no records")
    audit = {
        "selection_rule": "Resolved image license URL is http(s)://creativecommons.org/licenses/by/2.0/",
        "input_records": len(originals), "retained_records": len(selected),
        "excluded_records": len(exclusions),
        "input_unique_images": len({row["image"] for row in originals}),
        "retained_unique_coco_images": len({item["image"]["id"] for item in selected}),
        "excluded_reasons": dict(sorted(Counter(item["reason"] for item in exclusions).items())),
        "before": distribution(originals), "after": distribution([item["row"] for item in selected]),
        "exclusions": exclusions,
    }
    return selected, audit


def convert_jsonl(path, output_dir, *, coco_annotations, image_root, revision, config, split):
    """Write one portable, evaluation-only subset after all source validation succeeds."""
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    if config not in CONFIGS or split not in SPLITS:
        raise ValueError("config must be random/zeroshot and split must be dev/test")
    path, output_dir, image_root = Path(path), Path(output_dir), Path(image_root).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    coco_annotations = [Path(value) for value in coco_annotations]
    selected, audit = select_records(path, coco_annotations)
    if not selected:
        raise ValueError("no records have the requested CC BY 2.0 image license")
    from PIL import Image

    rows, media, verified = [], {}, {}
    for item in selected:
        source = (image_root / item["location"]).resolve()
        try:
            source.relative_to(image_root)
        except ValueError as error:
            raise ValueError(f"image path escapes image root: {source}") from error
        if not source.is_file():
            raise FileNotFoundError(f"missing selected image: {source}")
        if source not in verified:
            with Image.open(source) as image:
                image.verify()
            with Image.open(source) as image:
                image.load()
            verified[source] = digest(source)
        sha256 = verified[source]
        uri = f"media/{sha256}.jpg"
        media.setdefault(uri, source)
        original, image = item["row"], item["image"]
        converted = record(
            SOURCE, f"{revision}/{config}/{split}/{item['input_line']}", SPLITS[split], STATE,
            {"statement": {"type": "noul", "instructions": original["caption"],
                           "label": original["label"] == 1,
                           "src": f"vsr_{original['relation'].replace(' ', '_')}"}},
            media=[{"type": "image", "uri": uri}], group_id=f"vsr/media-{sha256}",
            media_sha256=[sha256], repository=REPOSITORY, revision=revision,
            config=config, upstream_split=split, input_line=item["input_line"],
            coco_image_id=image["id"], coco_url=image["coco_url"], flickr_url=image["flickr_url"],
            relation=original["relation"], annotation_license="Apache-2.0", image_license=item["license"],
        )
        materialize(converted)
        rows.append(converted)
    manifest = {
        "name": f"vsr-{config}-{split}-cc-by-2.0-v1",
        "description": "COCO-metadata license-filtered VSR subset; not a legal clearance or full benchmark",
        "base_revisions": {}, "trainable_sources": [], "eval_only_sources": [SOURCE], "holdout_sources": [],
        "sources": {SOURCE: {
            "repository": REPOSITORY, "revision": revision, "config": config, "upstream_split": split,
            "annotation_license": "Apache-2.0", "input_file": path.name, "input_sha256": digest(path),
            "coco_metadata_license": "CC-BY-4.0",
            "coco_annotations": [{"input_file": value.name, "sha256": digest(value)} for value in coco_annotations],
        }},
        "selection": audit, "unique_media_sha256": len(media),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "media").mkdir()
    artifacts = {}
    for uri, source in media.items():
        target = output_dir / uri
        shutil.copyfile(source, target)
        expected = uri.removeprefix("media/").removesuffix(".jpg")
        if digest(target) != expected:
            raise ValueError(f"image changed after validation: {source}")
        artifacts[uri] = {"sha256": expected}
    output = output_dir / f"{SPLITS[split]}.jsonl"
    write_jsonl(output, rows)
    manifest["files"] = {output.name: {"records": len(rows), "questions": len(rows), "sha256": digest(output)}}
    manifest["artifacts"] = artifacts
    write_json(output_dir / "manifest.json", manifest)
    validate_suite(output_dir, allow_test=split == "test")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="local original VSR UTF-8 JSONL")
    parser.add_argument("--coco-annotations", nargs="+", required=True, help="local COCO JSON image/license tables")
    parser.add_argument("--image-root", required=True, help="local root containing train2017/val2017 directories")
    parser.add_argument("--out", required=True, help="new output suite directory")
    parser.add_argument("--revision", required=True, help="source revision used to obtain VSR annotations")
    parser.add_argument("--config", required=True, choices=CONFIGS)
    parser.add_argument("--split", required=True, choices=tuple(SPLITS))
    args = parser.parse_args(argv)
    try:
        output = convert_jsonl(args.path, args.out, coco_annotations=args.coco_annotations,
                               image_root=args.image_root, revision=args.revision, config=args.config, split=args.split)
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
