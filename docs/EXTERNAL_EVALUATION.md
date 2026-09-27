# JevBench and Kev evaluation

The frozen recipe evaluates both released JevAny checkpoints and all 14 distinct
Kev checkpoints available through the main repositories and release tags on
September 27, 2026. Identical release aliases share a result. Unreleased
development branches are listed separately in the recipe.

The evaluation includes all public nonempty Kev development and test partitions,
the external SemIf, scienthoon, WANLI, TypeSafe, and ekzhang MMLU-Pro panels,
binding diagnostics, and JevBench's public easy, original, and hard tiers.
Historical versions retain separate reports. Training and calibration partitions
are excluded. Night-2 diagnostics are included with an explicit warning that
later Kev checkpoints trained on them.

Some Kev evaluation partitions are deliberately private. Their names, expected
counts, hashes, and mirror revisions remain in `manifest.json` under
`unavailable`; they cannot be reproduced from the public repository. JevBench's
231 public decisions do not include its sealed items. Public scores are not its
full leaderboard composite.

## Reproduce

Use Python 3.12 and the local inference dependencies:

```bash
python -m pip install -e '.[local,multimodal]'
python scripts/build_external_eval.py \
  --sources data/external-sources --download \
  --out data/external-eval --allow-test
```

This reads the exact revisions in
[`recipes/decision-evaluation.json`](../recipes/decision-evaluation.json).
The explicit `--allow-test` enables the fixed test partitions; do not tune or
select checkpoints using these results. No sampling is performed. Original
file hashes and record counts are checked before conversion.

Run one model on a local GPU:

```bash
PYTHONPATH=data/external-sources/kev:$PYTHONPATH \
python -m jevany.external_eval \
  --suite data/external-eval --model jevany-27b-sft \
  --out runs/external-eval/jevany-27b-sft/shard-0
```

Repeat for each model ID in the recipe. Kev uses its pinned upstream
implementation, loaded from `PYTHONPATH`. Each checkpoint retains its shipped
temperature and default evaluation precision: fp32 for the smaller Kev models,
and the trained bf16 backbone for the 27B models. `--dtype` is an explicit
experimental override and is recorded in the run.

For multiple GPUs, give each process its own GPU and output directory. Use
`--num-shards N --shard-index I` to divide a model's queue. Resume with the same
arguments: completed predictions and rejections are retained, and only a torn
final JSONL write is repaired. Changing the model, source, suite, precision,
temperature, or shard assignment requires a new output directory.

Generate reports after all processes finish:

```bash
python scripts/report_external_eval.py \
  --suite data/external-eval --sources data/external-sources \
  --runs runs/external-eval --out runs/external-report
```

The output includes detailed JSON reports, `scores.csv` with one row per
model/panel and explicit metric names, and a wide `accuracy.csv`. Incomplete
panels have empty accuracy cells in both tables. Published official Jev
baselines have separate model IDs and a source column. Local model latency is
reported separately from the predictor's wall time; hosted reference latencies
are not copied into local timing columns.

## What the scores mean

Requests contain no labels or reference distributions. Identical ordered
requests with identical context limits share inference, while each original
panel keeps its own labels, membership, and denominator. Option permutations
remain distinct. Context limits come from each Kev manifest; JevBench uses the
8,192-token state/row and 16,384-token packed context. Inputs are not truncated.

`all_requested_accuracy` counts rejected or missing knowable questions as wrong.
`answered_clean` reports accuracy, NLL, Brier, ECE, selective coverage, and ordinal
metrics only where a model returned a valid distribution. Missing predictions
mark a report incomplete. Unknowable questions measure confidence, not accuracy.
Raw, uncalibrated probability metrics are reported separately when logits and
the shipped temperature are available.

JevBench also uses its pinned native scorer, including its probability-sum
tolerance, lexicographic tie rule, ordinal metrics, and family summaries.
TypeSafe reports equal-case modal agreement and total-variation distance to the
reference distribution, with both all-row and answered-row values in `scores.csv`.
These metrics should not be replaced by ordinary question-weighted accuracy.

Published official Jev results are copied from Kev's committed reports with
their source paths, hashes, API model identity, and measurement dates. They are
marked `published_by_Kev_not_rerun`. Matching normally requires an original
manifest hash; historical aliases require identical partition bytes and context.
Scienthoon's converted rows instead verify every question ID, ordered option
list, and label, with this weaker match recorded explicitly. The API alias may
not expose a provider revision. Unmatched results remain references, rather
than being assigned to a different panel.

JevBench also publishes Jev 1.13.0's per-item public outcomes. Those provide a
separate public-tier accuracy reference, marked `published_by_JevBench_not_rerun`.
They contain no probability distributions, so no calibration metrics are
invented from them.

Latency is local model time on the recorded hardware; cloud price and the
JevBench speed/cost composite are not inferred. Existing JevAny multimodal and
interactive results remain documented in [EVALUATION.md](EVALUATION.md).
