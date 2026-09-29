<p align="center">
  <img src="docs/title.png" alt="JevAny: Your Jev from Any Model to Any Application" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/collections/tianxinwei/jevany-adaptive-decision-systems-6ab2c941bcecb4d2c61d1326"><img alt="Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97-checkpoints-ffb000"></a>
  <a href="docs/API.md"><img alt="API docs" src="https://img.shields.io/badge/docs-API-0ea5e9"></a>
  <a href="docs/CASES.md"><img alt="Examples" src="https://img.shields.io/badge/examples-gallery-8b5cf6"></a>
  <a href="pyproject.toml"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="https://github.com/weitianxin/JevAny/actions"><img alt="Tests" src="https://github.com/weitianxin/JevAny/actions/workflows/ci.yml/badge.svg"></a>
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
git clone https://github.com/weitianxin/JevAny.git
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

### Pointer vs direct-token

Both variants perform one backbone prefill and return a probability distribution
without generating answer text, so inference speed should be similar on comparable
inputs. Pointer learns a compact head over the decision marker and option
representations; it supports more than 255 choices, bounded in practice by the
context window. Direct-token has no specialized decision head: it prefixes each
option with one of 255 fixed, single-token labels and scores those labels with the
base LM head. The released direct-token model has the better 4B JevBench result,
but its full-vocabulary cross-entropy makes training slower and it is limited to
255 choices.

### Release roadmap

The five LoRA checkpoints are the current release. Full-parameter SFT, improved
post-training, and hyperparameter-optimized variants are planned as later
releases; they are not included in this version.

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

## Evaluation

Accuracy is reported on two fixed protocols: Transfer-v9's 1,046 clean,
knowable decisions and all 231 public JevBench development items. Every local
run had complete coverage with no rejected or truncated records. JevBench public
accuracy is a development-set diagnostic, not the official sealed Benchmark
Heaven score.

| Released model | Readout | Transfer-v9 | JevBench Acc. | NLL ↓ | Brier ↓ | ECE ↓ |
|---|---|---:|---:|---:|---:|---:|
| Gemma 4B LoRA | Pointer | 70.84% | 77.49% | 0.536 | 0.309 | 0.043 |
| Qwen3.5 4B LoRA | Pointer | **78.68%** | 80.09% | 0.455 | 0.259 | 0.037 |
| Qwen3.5 4B Direct-Token LoRA | Direct-token | 78.20% | **80.95%** | 0.433 | 0.256 | 0.051 |
| Qwen3.8 27B LoRA | Pointer | **85.76%** | **90.48%** | **0.270** | **0.145** | 0.036 |
| Muse Glimmer 30B LoRA | Pointer | 83.46% | 87.45% | 0.316 | 0.174 | **0.027** |

For context, the same two public protocols give:

| Reference model | Transfer-v9 | JevBench public | Provenance |
|---|---:|---:|---|
| Kev-4B | 74.19% | 75.32% | Local public-checkpoint runs |
| Jev 1.13.0 | 85.37% | 86.58% | Local API run / published JevBench reference |
| Laya (`55cf4c4`) | 52.29% | 58.01% | Local pinned-revision runs |

The direct-token 4B model leads the released 4B models on JevBench, while the
pointer 4B model is slightly better on Transfer-v9. Reference comparisons use
accuracy only; calibration metrics are reported for released models where the
same local protocol produced them. No unavailable sealed-test scores are mixed in.

[Machine-readable release results](results/model-family-v2.json) ·
[Evaluation protocols and historical results](docs/EVALUATION.md).

## Training

The current family was trained on **1,772,725 text records / 2,180,242 labelled
decisions** spanning preference, agent/tool decisions, reasoning, classification,
and safety. Detailed mixture and source-level composition are not released.

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
