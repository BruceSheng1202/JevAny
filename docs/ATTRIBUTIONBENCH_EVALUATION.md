# AttributionBench Evaluation

This standalone converter makes [AttributionBench](https://huggingface.co/datasets/osunlp/AttributionBench)
usable with JevAny's existing evaluation interfaces. It does not train a model,
change default benchmarks, or run inference during conversion.

## Task

Each original row becomes one `noul` question: is the entire `claim` supported
by its complete `references` bundle? The model receives those two fields and
an instruction to judge only the supplied evidence.

The original labels map exactly as follows:

- `attributable` becomes `true`.
- `not attributable` becomes `false`.

Claim and reference strings, reference order, and repeated references are
preserved. Empty reference lists retain their original annotations; the
converter does not repair labels. Original responses, labels, raw claims, and
citation links are not added to model evidence. The task is not per-reference
ranking, full-response correctness, or end-to-end retrieval.

## Source And License

The documented source is `osunlp/AttributionBench`, pinned to revision
`62569e644f4186606f54f742178a4517431b42e1`. Its dataset card declares
Apache-2.0. Review and retain the notices and terms of constituent sources;
this converter does not grant additional rights to those texts.

Download original exports separately. Conversion itself needs no network,
Hugging Face libraries, tokenizer, or model. Do not commit downloaded corpora
or generated suites. For example, after creating a local export directory:

```bash
curl --fail --location \
  https://huggingface.co/datasets/osunlp/AttributionBench/resolve/62569e644f4186606f54f742178a4517431b42e1/dev_all_subset_balanced.jsonl \
  --output /path/to/exports/attributionbench/dev_all_subset_balanced.jsonl
```

All balancing configurations share the same evaluation files. They are not
independent evaluation panels and must not be counted multiple times.

## Convert A Local Export

Use a base JevAny installation with Python 3.12 or newer. Run the module from
the checkout root; paths below are placeholders for local absolute paths:

```bash
cd /path/to/JevAny
python -m scripts.convert_attributionbench \
  /path/to/exports/attributionbench/dev_all_subset_balanced.jsonl \
  --out /path/to/suites/attributionbench-dev \
  --revision 62569e644f4186606f54f742178a4517431b42e1 \
  --split dev
```

The required original fields are `id`, `src_dataset`, `claim`, `references`,
and `attribution_label`. The first three must be non-empty strings;
`references` must be a list of strings. An optional `question` must be a
string or null. Unknown labels, malformed rows, duplicate IDs, and empty
exports are rejected with physical input line numbers. All rows are validated
before creating output. Existing output directories are never overwritten.

Original `dev` maps to `development`. Original `test` and `test_ood` each map
to the locked `test` partition in a separate output directory. Supply the
matching original export and `--split` value; the converter does not infer or
verify an export's upstream split from its contents. It does not accept
training splits or resplit, shuffle, truncate, or relabel records.

The output is one JSONL partition and a manifest, for example
`/path/to/suites/attributionbench-dev/development.jsonl` and
`/path/to/suites/attributionbench-dev/manifest.json`. The manifest records the
input revision/checksum, original split, output checksum/counts, and an
evaluation-only source policy. Record metadata retains original IDs, source
datasets, original splits, revision, and physical input lines. Model requests
exclude labels and provenance.

## Validate And Evaluate

Development data can be checked and scored with the existing commands:

```bash
jevany data check-suite /path/to/suites/attributionbench-dev
jevany eval --run /path/to/checkpoint \
  --suite /path/to/suites/attributionbench-dev \
  --out /path/to/evaluations/attributionbench-dev
```

For a separately converted original `test` or `test_ood` export, explicitly
unlock its test partition:

```bash
jevany data check-suite /path/to/suites/attributionbench-test --allow-test
jevany eval --run /path/to/checkpoint \
  --suite /path/to/suites/attributionbench-test --allow-test \
  --out /path/to/evaluations/attributionbench-test
```

Evaluation output directories must also be new. Freeze the checkpoint,
readout, prompt, source revision, and partition before comparing models.
The task names retain each original `src_dataset` for per-source reporting.

## Limits

Related records are grouped by source dataset and original question when it
is nonblank. Without a question, the grouping uses that source dataset and
an exact reference-bundle hash. Group IDs do not depend on the output split.
This is a conservative grouping convention, not a complete semantic
contamination audit. It does not detect paraphrases or every shared document.

`check-suite` compares only partitions in the same manifest. It does not
compare separate conversion directories. Creating independent development,
test, or OOD suites does not establish that they are disjoint. Audit overlap
before using any data for selection or training, and keep test data locked.
The evaluation-only manifest policy is not a security boundary against
someone deliberately loading raw records into a separate training workflow.

Conversion validates schema and provenance, not model token limits. Frozen
suite scoring aborts on overlong inputs rather than silently dropping them.
For custom-data scoring that records context rejections and coverage, use
`--data /path/to/suites/attributionbench-dev/development.jsonl` instead of
`--suite`; report requested/evaluated counts and count rejected questions as
wrong in any all-request headline accuracy. Do not claim complete coverage
from answered-only accuracy.

The public release training corpus is not fully disclosed. These results
would measure external attribution performance, not prove contamination-free
generalization or improved end-to-end RAG quality.
