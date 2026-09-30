<p align="center">
  <img src="docs/title.png" alt="JevAny: Your Jev from Any Model to Any Application" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/collections/tianxinwei/jevany-adaptive-decision-systems-6ab2c941bcecb4d2c61d1326"><img alt="Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97-checkpoints-ffb000"></a>
  <a href="docs/API.md"><img alt="API docs" src="https://img.shields.io/badge/docs-API-0ea5e9"></a>
  <a href="docs/CASES.md"><img alt="Examples" src="https://img.shields.io/badge/examples-gallery-8b5cf6"></a>
  <a href="pyproject.toml"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="https://github.com/SimpleJev/JevAny/actions"><img alt="Tests" src="https://github.com/SimpleJev/JevAny/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-32d6c5"></a>
</p>

<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <strong>One decision API. Five open LoRA checkpoints. Any bounded choice.</strong><br>
  Turn state, questions, and candidate answers into calibrated choices—without generating answer text.
</p>

<p align="center">
  <img src="docs/hero.png" alt="JevAny workflow: train a Jev model with multi-modal data and RLCR/SFT, then deploy through a unified API with test environments and practical examples" width="100%">
</p>

| 🤗 [Models](#pretrained-models) | 📊 [Results](#evaluation) | ⚡ [Serve](#inference--serving) | 🛠️ [Train](#training) |
|---|---|---|---|
| Gemma, Qwen, Muse | Transfer + JevBench | Python + HTTP | SFT + RLCR |

## Demos

Examples recorded with an earlier compatible JevAny checkpoint:

[![JevAny choosing actions across robotics, browser, software, laboratory and mobility tasks](docs/demos/jevany-cases.gif)](docs/CASES.md)

[Explore 30 selected successful runs](docs/CASES.md), or try your own model with the [examples and test environments](#examples--test-environments).

## Installation

Use Python 3.12 or newer. Clone the repository and create an environment:

```bash
git clone https://github.com/SimpleJev/JevAny.git
cd JevAny
python3.12 -m venv .venv
source .venv/bin/activate
```

Choose the dependencies for your use case:

| Use case | Install |
|---|---|
| Call an existing HTTP server | `python -m pip install -e .` |
| Train a text model | `python -m pip install -e '.[train]'` |
| Run a text model locally or serve it over HTTP | `python -m pip install -e '.[serve]'` |
| Run a released model with native media support | `python -m pip install -e '.[serve,multimodal]'` |

The client-only installation does not install PyTorch. For image/video training, use `.[train,multimodal]`. Run the commands below from the repository root; model-specific hardware requirements are listed under [Pretrained Models](#pretrained-models).

## Pretrained Models

| Model | Readout | Intended use |
|---|---|---|
| [JevAny-Gemma-4B-LoRA](https://huggingface.co/tianxinwei/JevAny-Gemma-4B-LoRA) | Pointer | Compact Gemma release |
| [JevAny-Qwen3.5-4B-LoRA](https://huggingface.co/tianxinwei/JevAny-Qwen3.5-4B-LoRA) | Pointer | Compact, flexible choice count |
| [JevAny-Qwen3.5-4B-Direct-Token-LoRA](https://huggingface.co/tianxinwei/JevAny-Qwen3.5-4B-Direct-Token-LoRA) | Direct-token | Best released 4B JevBench accuracy |
| [JevAny-Qwen3.8-27B-LoRA](https://huggingface.co/tianxinwei/JevAny-Qwen3.8-27B-LoRA) | Pointer | Default; highest released accuracy |
| [JevAny-Muse-Glimmer-30B-LoRA](https://huggingface.co/tianxinwei/JevAny-Muse-Glimmer-30B-LoRA) | Pointer | Muse Glimmer alternative |

These are LoRA adapters: the corresponding base model is loaded separately and
its license and access terms apply. Allow roughly twice the base parameter count
in bytes for BF16 weights, plus runtime memory. See the
[hardware and loading guide](docs/DEPLOYMENT.md#checkpoints-and-hardware).

### Training compute

| Model | Released step | Parallel GPUs | Wall time | GPU-hours |
|---|---:|---:|---:|---:|
| Gemma 4B LoRA | 2,771 | 32 H200 | ~3.28 h | ~104.9 |
| Qwen3.5 4B LoRA | 13,850 | 32 H200 | 10.33 h | 330.6 |
| Qwen3.5 4B Direct-Token LoRA | 9,695 | 32 H200 | 8.41 h | 269.1 |
| Muse Glimmer 30B LoRA | 3,324 | 40 H200 | 2.88 h | 115.4 |
| Qwen3.8 27B LoRA | 22,160 | 32 H200 | 18.83 h | 602.7 |

The five released checkpoints represent approximately **1,423 H200 GPU-hours**
of training, with at most 40 GPUs used in parallel within one run. GPU-hours are
elapsed training time through the released checkpoint multiplied by the DDP
world size; ablations, evaluation, and training after a selected checkpoint are
excluded. Gemma uses run/checkpoint timestamps because its earlier checkpoint
format did not store cumulative elapsed seconds; the other figures come from
checkpoint or terminal trainer telemetry.

### Pointer vs direct-token

- **Pointer:** a compact learned head scores the decision marker against option
  representations. It trains efficiently and supports more than 255 choices,
  subject to the context window.
- **Direct-token:** the base LM head scores 255 fixed single-token option labels.
  It leads the released 4B models on JevBench, but full-vocabulary training is
  slower and requests are limited to 255 choices.
- **Inference:** both use one backbone prefill and no answer generation, so
  latency should be similar for comparable inputs.

### Release roadmap

- **Now:** five verified LoRA checkpoints.
- **Next:** full-parameter SFT.
- **Later:** improved post-training and hyperparameter-optimized variants.

## Evaluation

**Transfer** is a fixed cross-domain and robustness evaluation over 1,046 clean,
knowable decisions. Its item-level mean spans Emotion, PAWS, QNLI, TweetEval,
MMLU, MMLU-Pro, SciQ, and four robustness slices. **JevBench** is accuracy across
all 231 public development items, not the sealed leaderboard score. NLL, Brier,
and ECE in the main table are Transfer metrics; every run covers every item.

| Model | Transfer ↑ | JevBench ↑ | NLL ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|
| Kev-4B | 74.19% | 75.32% | 0.858 | 0.380 | 0.125 |
| Kev-27B (`01b8199`) | 82.31% | 85.28% | 0.533 | 0.265 | 0.050 |
| Jev 1.13.0 | 85.37% | 86.58% | 0.644 | 0.212 | 0.033 |
| Laya (`55cf4c4`) | 52.29% | 58.01% | 1.264 | 0.615 | 0.127 |
| **JevAny releases** |  |  |  |  |  |
| Gemma 4B LoRA | 70.84% | 77.49% | 0.706 | 0.369 | 0.056 |
| Qwen3.5 4B LoRA | 78.68% | 80.09% | 0.587 | 0.297 | 0.035 |
| Qwen3.5 4B Direct-Token LoRA | 78.20% | 80.95% | 0.564 | 0.291 | **0.029** |
| Muse Glimmer 30B LoRA | 83.46% | 87.45% | 0.464 | 0.229 | 0.032 |
| **Qwen3.8 27B LoRA** | **85.76%** | **90.48%** | **0.392** | **0.200** | 0.030 |

### Transfer breakdown

Columns are dataset or robustness-slice accuracy. Sample counts are respectively
80 / 80 / 80 / 80 / 80 / 200 / 80 / 80 / 96 / 80 / 110.

| Model | Emo | PAWS | QNLI | Tweet | MMLU | M-Pro | SciQ | Buried | Comp. | Policy | Ctrl. | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baselines** |  |  |  |  |  |  |  |  |  |  |  |  |
| Kev-4B | 53.75 | 73.75 | 92.50 | 73.75 | 70.00 | 50.00 | 98.75 | 66.25 | 92.71 | 77.50 | 92.73 | 74.19 |
| Kev-27B | 60.00 | 78.75 | 95.00 | 80.00 | 83.75 | 66.00 | 97.50 | 76.25 | 89.58 | 96.25 | 99.09 | 82.31 |
| Jev 1.13.0 | 58.75 | 80.00 | 91.25 | 81.25 | 90.00 | 84.00 | 98.75 | 70.00 | 90.62 | 97.50 | 94.55 | 85.37 |
| Laya | 61.25 | 77.50 | 83.75 | 75.00 | 33.75 | 11.00 | 87.50 | 58.75 | 51.04 | 52.50 | 47.27 | 52.29 |
| **JevAny (Ours)** |  |  |  |  |  |  |  |  |  |  |  |  |
| Gemma 4B | 75.00 | 80.00 | 91.25 | 80.00 | 65.00 | 38.50 | 97.50 | 70.00 | 67.71 | 77.50 | 81.82 | 70.84 |
| Qwen 4B P | 86.25 | 75.00 | 93.75 | 81.25 | 77.50 | 58.50 | 97.50 | 77.50 | 87.50 | 76.25 | 81.82 | 78.68 |
| Qwen 4B DT | 85.00 | 77.50 | 95.00 | 81.25 | 75.00 | 52.50 | 98.75 | 78.75 | 86.46 | 83.75 | 81.82 | 78.20 |
| Muse 30B | 81.25 | 77.50 | 92.50 | 83.75 | 83.75 | 61.00 | 98.75 | 82.50 | 95.83 | 98.75 | 90.91 | 83.46 |
| **Qwen 27B** | **90.00** | **86.25** | **95.00** | **83.75** | **86.25** | **68.00** | **97.50** | **78.75** | **90.62** | **97.50** | **92.73** | **85.76** |

`Buried` tests hidden instructions; `Comp.` tests AND/OR/conditional composition;
`Policy` contrasts authorization/deadline rules; `Ctrl.` contains 11 knowable
policy controls. Values are percentages.

### JevBench breakdown

| Model | Easy | Original | Hard | Score |
|---|---:|---:|---:|---:|
| **Baselines** |  |  |  |  |
| Kev-4B | 100.00% | 94.44% | 52.25% | 75.32% |
| Kev-27B | 100.00% | 100.00% | 69.37% | 85.28% |
| Jev 1.13.0 | 100.00% | 98.61% | 72.97% | 86.58% |
| Laya | 95.83% | 70.83% | 33.33% | 58.01% |
| **JevAny (Ours)** |  |  |  |  |
| Gemma 4B | 100.00% | 95.83% | 55.86% | 77.49% |
| Qwen 4B P | 100.00% | 95.83% | 61.26% | 80.09% |
| Qwen 4B DT | 100.00% | 98.61% | 61.26% | 80.95% |
| Muse 30B | 100.00% | 97.22% | 75.68% | 87.45% |
| **Qwen 27B** | **100.00%** | **98.61%** | **81.08%** | **90.48%** |

The direct-token 4B model leads the released 4B models on JevBench, while the
pointer 4B model is slightly better on Transfer. Kev and Laya use complete local
public-checkpoint runs (Kev-27B pinned to `01b8199`, Laya to `55cf4c4`); Jev uses
a complete local API Transfer run and JevBench's published per-tier accuracy.

### What we ablated

- **Pointer structure:** linear, MLP, and zero-initialized residual heads; the
  residual head won the controlled screen on development NLL and calibration.
- **Representation readout:** decision-token, query-mean, option-mean, and
  combined variants. Query-mean led the early screen; the release recipe uses
  decision-marker / option-close after the full model-family run.
- **Loss:** cross-entropy, pure InfoNCE, and mixed objectives; CE gave the best
  Transfer accuracy in the loss sweep, while small contrastive terms mainly
  improved calibration. The released checkpoints use CE.
- **Readout family:** at 4B, direct-token improves JevBench (80.95% vs 80.09%),
  while pointer is slightly stronger on Transfer (78.68% vs 78.20%).

### Reproducibility

Transfer reports 1,046 scored decisions with no missing examples; JevBench reports
all 231 public development items. Exact suite hashes and unrounded metrics are in
the machine-readable results. Release manifests, checkpoint-native reload reports,
and GPU loader/reconstruction parity were checked before publishing.

[Machine-readable release results](results/model-family-v2.json) ·
[Evaluation protocols and historical results](docs/EVALUATION.md) ·
[Method and ablation report](docs/JEVANY_METHOD_AND_ABLATIONS.pdf)
([LaTeX source](docs/JEVANY_METHOD_AND_ABLATIONS.tex)).

## Inference & Serving

### Python API

With an [HTTP server](#http-server) running, send a state and a question with named options. The answer contains the selected option and each option's probability:

```python
from jevany import Choice, JevClient

state = {"ticket": "I was charged twice. Please help."}
questions = {
    "department": Choice(
        instructions="Which team should handle this?",
        criteria={"billing": "Payment problems", "shipping": "Delivery problems"},
    ),
}

jev = JevClient("http://127.0.0.1:8008")
result = jev.system_one(state=state, questions=questions)
answer = result["answers"]["department"]
print(answer["choice"])
print(answer["probabilities"])
```

Use `Noul` for binary questions and `Score` for ordered levels. The [API reference](docs/API.md) describes all three question types and the Jev-compatible request/answer format.

### HTTP Server

For the default released model, install `.[serve,multimodal]` and use hardware that meets the [27B requirements](#pretrained-models):

```bash
jevany serve --checkpoint tianxinwei/JevAny-Qwen3.8-27B-LoRA \
  --device cuda --dtype bf16 --port 8008
```

To serve the smaller model from the SFT example instead:

```bash
jevany serve --checkpoint runs/my-jev --model-name my-jev --port 8008
```

### In-Process Inference

Load a checkpoint once in your application and reuse `state` and `questions` from the example above:

```python
from jevany import JevModel

jev = JevModel.from_pretrained("runs/my-jev", model_name="my-jev")
result = jev.system_one(state=state, questions=questions)
```

See the [deployment guide](docs/DEPLOYMENT.md) for more ways to call a model and [media setup](docs/DEPLOYMENT.md#native-media-and-limits) for image and video inputs.

## Examples & Test Environments

Open the playground in your local browser. The included replays need no GPU, model download, or inference server:

```bash
python -m pip install -e .
jevany demo
```

These GIFs show accelerated replays from an earlier compatible JevAny checkpoint. Each replay preserves the model's actual choices and original option probabilities.

### [Robot peg insertion](examples/README.md#robot-peg-insertion)

Use a Franka gripper to grasp, align and insert a peg, checked by PyBullet contact physics.

![Robot browser replay showing the Franka arm inserting a peg, recorded model probabilities and physical success checks](docs/demos/playground-arm.gif)

### [Doom corridor · 3D](examples/README.md#doom-corridor-3d)

Start in the final room, kill the enemies on the left and right, then move forward through the cleared room. Uses ViZDoom and the included Freedoom assets.

![Doom checkpoint replay: kill both enemies, then advance](docs/demos/playground-doom.gif)

### [Crafter survival · 2D](examples/README.md#crafter-survival-2d)

Gather wood, craft tools and mine stone while managing health and supplies.

![Crafter browser replay showing resource gathering, crafting actions and progress through four goal milestones](docs/demos/playground-crafter.gif)

### Live control

Install the optional game engines to play yourself, or connect a [running model server](#http-server) and choose **Run model** in the browser:

```bash
python -m pip install -e '.[demo]'
jevany demo --base-url http://127.0.0.1:8008 --text-only
```

Live model runs currently use text state. Robot control uses the separate `.[robotics]` extra. See the [playground guide](examples/README.md) for setup, platform requirements and environment APIs, or [integrations](docs/INTEGRATIONS.md) to combine Jev decisions with an LLM planner.

## Training

The current family was trained on **1,772,725 text records / 2,180,242 labelled
decisions** spanning preference, agent/tool decisions, reasoning, classification,
and safety.

Prepare the included starter data:

```bash
jevany data init --out data/starter
jevany data validate data/starter/train.jsonl
```

### SFT

```bash
jevany train --config recipes/sft.toml --dry-run
jevany train --config recipes/sft.toml
```

Use your own JSONL with `--data`, or continue a released model with
[`recipes/finetune.toml`](recipes/finetune.toml).

### RLCR

Continue an SFT checkpoint with correctness-and-calibration rewards:

```bash
jevany train --config recipes/rlcr.toml
```

[Training guide](docs/TRAINING.md) · [Data format](docs/DATA.md) ·
[RLCR objective](docs/ALGORITHM.md#rlcr)

## Supported Model Families

![26 supported models across Qwen, Gemma, Muse, Mistral, GLM, Nemotron and Llama](docs/supported-model-families.svg)

[Model IDs, supported inputs and setup requirements](docs/TRAINING.md#backbone-support).

## Documentation and Contributing

[Training](docs/TRAINING.md) · [Deployment](docs/DEPLOYMENT.md) · [API compatibility](docs/API.md) · [Data](docs/DATA.md) · [Evaluation](docs/EVALUATION.md) · [Contributing](CONTRIBUTING.md)

JevAny is independent of Jev and TypeSafe and includes no Jev weights or private implementation. It includes infrastructure adapted from [Kev](https://github.com/jaredpalmer/kev); see [NOTICE](NOTICE) and [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md). Code and starter data are Apache-2.0. Base models and upstream datasets retain their own terms.
