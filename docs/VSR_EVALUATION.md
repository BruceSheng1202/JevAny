# VSR License-Filtered Evaluation

This standalone offline converter prepares a Visual Spatial Reasoning (VSR)
subset using images whose COCO metadata identifies **CC BY 2.0**. It does not
train a model, change default benchmarks, download images, or run inference.

Call the output a **COCO-metadata CC BY 2.0-filtered VSR subset**, not the full
VSR benchmark or a legally cleared image collection.

## Task and Source

An original VSR caption becomes one `noul` question about the supplied image.
The converter preserves the caption exactly and maps integer `0` to `false`,
integer `1` to `true`. Labels, relation categories, votes, object metadata,
source identifiers and licenses do not enter model evidence. It does not
rewrite captions, generate negative examples, or repair annotations.

Use the [author repository](https://github.com/cambridgeltl/visual-spatial-reasoning),
starting with revision `b27a0af0ee1462d2b6b92c8c83e869d9254a241a`.
Its original configurations are `random` and `zeroshot`, with `train`, `dev`
and `test` partitions. This converter accepts only `dev` and `test` and keeps
those partitions separate. The author provides public test labels.

The Hugging Face loading script and older dataset-card statistics need not
match this author revision. Pin the original export rather than mixing
versions. Record the revision actually used; declaring it is not proof that
a local file was obtained from that revision.

## Licensing and Attribution

These are three different licensing layers:

- The VSR project and dataset card declare Apache-2.0. Retain applicable
  upstream notices, cite the source, and describe the conversion when
  distributing VSR annotations or derived records.
- The [COCO Consortium's annotations and website](https://cocodataset.org/#termsofuse)
  are licensed under CC BY 4.0; retain the appropriate COCO attribution.
- Individual photographs retain their authors' rights and conditions.
  The filter accepts only COCO image records whose resolved license URL is
  `http://creativecommons.org/licenses/by/2.0/` or its HTTPS equivalent.
  It does not infer permission from a license ID or display name. Other
  licenses are excluded from this particular policy, not declared invalid.

CC BY 2.0 has attribution and other conditions. Review the original source
and retain required author, source and license information for the intended
use. The converter preserves available Flickr URLs and COCO license records,
but it does not retrieve photographer names, verify a license grant, check
current Flickr pages, or grant rights. Follow your organization's data-use
requirements where applicable. Private evaluation is not automatically exempt
from image terms.

Do not commit real photographs, downloaded annotations or generated suites.
Committed tests use original synthetic images and annotation fixtures. If
sharing a generated suite or results containing photographs, separately
satisfy the applicable attribution and redistribution conditions. Do not
relabel all source material as Apache-2.0.

## Prepare Local Inputs

Obtain a fixed author JSONL export and COCO image/license tables separately.
The official [COCO download page](https://cocodataset.org/#download) links the
2017 train/validation annotation archive. Caption or instance annotation JSON
contains `images` and `licenses` tables. Supply both train and validation
metadata when the VSR export references both COCO partitions; consistent
repeated tables across files are supported.

For example, prepare these absolute local paths outside the checkout:

- `/tmp/vsr-input/random-dev.jsonl`
- `/tmp/vsr-input/captions_train2017.json`
- `/tmp/vsr-input/captions_val2017.json`
- `/tmp/vsr-input/images/train2017/` and `/tmp/vsr-input/images/val2017/`

Place only needed, appropriately usable images under their original twelve-digit
COCO filenames. Excluded images need not be present. Missing selected images,
unknown license mappings and contradictory metadata are errors, not silent
exclusions. `image_link`, COCO filename, image ID and original COCO partition
must agree. Symlinks resolving outside the image root are rejected.

Conversion needs Pillow, already included in JevAny's `dev` and `multimodal`
extras. Run commands from the repository checkout with JevAny installed:

```bash
python -m scripts.convert_vsr /tmp/vsr-input/random-dev.jsonl \
  --coco-annotations /tmp/vsr-input/captions_train2017.json /tmp/vsr-input/captions_val2017.json \
  --image-root /tmp/vsr-input/images \
  --revision b27a0af0ee1462d2b6b92c8c83e869d9254a241a \
  --config random --split dev --out /tmp/vsr-random-dev-cc-by

python -m jevany data check-suite /tmp/vsr-random-dev-cc-by
```

The output must not already exist. All annotations, associations and selected
images are validated before output creation. Images are copied byte-for-byte
into content-addressed media paths, without resizing or re-encoding. An I/O
failure during writing may leave an incomplete output; inspect it and use a
new output path rather than overwriting it.

The manifest records input SHA-256 values, the exact license selection rule,
retained/excluded counts and reasons, and before/after label and relation
counts. Record metadata retains source configuration, original partition and
line number, COCO ID, Flickr/COCO URLs, image license and media hash. Relative
media references make the suite portable. Hashes identify supplied bytes;
they do not establish the authenticity or legal status of the input metadata.

## Optional Evaluation

Conversion and synthetic tests require no model. Scoring requires a checkpoint
or endpoint that supports image inputs and Noul decisions; a text-only model
is not sufficient.

For a development suite, the existing evaluator can be used:

```bash
python -m jevany eval --run /path/to/compatible-image-checkpoint \
  --suite /tmp/vsr-random-dev-cc-by --out /tmp/vsr-dev-scores
```

The existing multimodal evaluation script also supports development media
controls (full, blank and shuffled). Its CLI loads `development` only. These
controls diagnose image sensitivity; replacing the image does not create
new ground-truth annotations.

For an original test export, convert with `--split test` into another fresh
directory, then explicitly validate the locked partition:

```bash
python -m jevany data check-suite /tmp/vsr-random-test-cc-by --allow-test
```

For test scoring, the generic evaluator selects the locked test partition when
`--allow-test` is supplied with `--suite`:

```bash
python -m jevany eval --run /path/to/compatible-image-checkpoint \
  --suite /tmp/vsr-random-test-cc-by --allow-test --out /tmp/vsr-test-scores
```

Without that flag, `--suite` selects development. Do not rename test to
development or treat raw JSONL loading as a suite lock.

## Interpretation Limits

Freeze the data revision, configuration, partition, filtering policy, prompt,
checkpoint and readout before scoring. Report the retained size, exclusions,
label/relation counts, rejected predictions and coverage. Do not select records
according to model performance or compare a filtered result as if it were the
full benchmark.

Multiple captions for one image remain distinct records but share a media-hash
group. Original random splits were not designed to guarantee image-disjoint
partitions. Zeroshot describes VSR's object-concept split, not proof that a
modern model has never seen those concepts or COCO pictures. The configurations
are not necessarily independent evaluation sets.

Each output contains one original partition. Suite validation does not compare
separate directories or audit undisclosed pretraining corpora. Compare media
hashes/groups across available prior data when making exposure claims; no such
claim follows from license filtering. This contribution supplies evaluation
infrastructure, not model scores or evidence of capability improvement.
