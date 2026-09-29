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

Accuracy uses two fixed protocols: Transfer-v9's 1,046 clean, knowable decisions
and all 231 public JevBench development items. Every run covered every item.
JevBench public is a development diagnostic, not the sealed leaderboard score.

| Model | Type | T-v9 Acc ↑ | T-v9 NLL ↓ | T-v9 Brier ↓ | T-v9 ECE ↓ | JB Acc ↑ | JB NLL ↓ | JB Brier ↓ | JB ECE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kev-4B | Reference | 74.19% | 0.858 | 0.380 | 0.125 | 75.32% | 0.537 | 0.310 | **0.021** |
| Jev 1.13.0 | Reference | 85.37% | 0.644 | 0.212 | 0.033 | 86.58% | —¹ | —¹ | —¹ |
| Laya (`55cf4c4`) | Reference | 52.29% | 1.264 | 0.615 | 0.127 | 58.01% | 0.927 | 0.533 | 0.095 |
| **JevAny releases** |  |  |  |  |  |  |  |  |  |
| Gemma 4B LoRA | Pointer | 70.84% | 0.706 | 0.369 | 0.056 | 77.49% | 0.536 | 0.309 | 0.043 |
| Qwen3.5 4B LoRA | Pointer | 78.68% | 0.587 | 0.297 | 0.035 | 80.09% | 0.455 | 0.259 | 0.037 |
| Qwen3.5 4B Direct-Token LoRA | Direct-token | 78.20% | 0.564 | 0.291 | **0.029** | 80.95% | 0.433 | 0.256 | 0.051 |
| Muse Glimmer 30B LoRA | Pointer | 83.46% | 0.464 | 0.229 | 0.032 | 87.45% | 0.316 | 0.174 | 0.027 |
| **Qwen3.8 27B LoRA** | **Pointer** | **85.76%** | **0.392** | **0.200** | 0.030 | **90.48%** | **0.270** | **0.145** | 0.036 |

¹ JevBench published only Jev 1.13's per-tier correct counts, not prediction
probabilities, so its JevBench NLL, Brier, and ECE cannot be computed.

### Transfer-v9 breakdown

| Dataset / robustness slice | n | Kev | Jev | Laya | Gemma 4B | Qwen 4B P | Qwen 4B DT | Muse 30B | Qwen 27B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Emotion | 80 | 53.75% | 58.75% | 61.25% | 75.00% | 86.25% | 85.00% | 81.25% | 90.00% |
| PAWS | 80 | 73.75% | 80.00% | 77.50% | 80.00% | 75.00% | 77.50% | 77.50% | 86.25% |
| QNLI | 80 | 92.50% | 91.25% | 83.75% | 91.25% | 93.75% | 95.00% | 92.50% | 95.00% |
| TweetEval offensive | 80 | 73.75% | 81.25% | 75.00% | 80.00% | 81.25% | 81.25% | 83.75% | 83.75% |
| MMLU | 80 | 70.00% | 90.00% | 33.75% | 65.00% | 77.50% | 75.00% | 83.75% | 86.25% |
| MMLU-Pro | 200 | 50.00% | 84.00% | 11.00% | 38.50% | 58.50% | 52.50% | 61.00% | 68.00% |
| SciQ | 80 | 98.75% | 98.75% | 87.50% | 97.50% | 97.50% | 98.75% | 98.75% | 97.50% |
| Buried instruction (Emotion/PAWS/QNLI/TweetEval) | 80 | 66.25% | 70.00% | 58.75% | 70.00% | 77.50% | 78.75% | 82.50% | 78.75% |
| Compositional holdouts (AND/OR/conditional) | 96 | 92.71% | 90.62% | 51.04% | 67.71% | 87.50% | 86.46% | 95.83% | 90.62% |
| Contrastive policy (authorization/deadline) | 80 | 77.50% | 97.50% | 52.50% | 77.50% | 76.25% | 83.75% | 98.75% | 97.50% |
| Knowable controls (11 policy tasks) | 110 | 92.73% | 94.55% | 47.27% | 81.82% | 81.82% | 81.82% | 90.91% | 92.73% |
| **Overall** | **1,046** | **74.19%** | **85.37%** | **52.29%** | **70.84%** | **78.68%** | **78.20%** | **83.46%** | **85.76%** |

### JevBench breakdown

| Tier | n | Kev | Jev | Laya | Gemma 4B | Qwen 4B P | Qwen 4B DT | Muse 30B | Qwen 27B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy | 48 | 100.00% | 100.00% | 95.83% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Original | 72 | 94.44% | 98.61% | 70.83% | 95.83% | 95.83% | 98.61% | 97.22% | 98.61% |
| Hard | 111 | 52.25% | 72.97% | 33.33% | 55.86% | 61.26% | 61.26% | 75.68% | 81.08% |
| **Overall** | **231** | **75.32%** | **86.58%** | **58.01%** | **77.49%** | **80.09%** | **80.95%** | **87.45%** | **90.48%** |

The direct-token 4B model leads the released 4B models on JevBench, while the
pointer 4B model is slightly better on Transfer-v9. Kev and Laya use complete
local public-checkpoint runs (Laya pinned to `55cf4c4`); Jev uses a complete
local API Transfer-v9 run and JevBench's published per-tier accuracy reference.
No unavailable sealed-test scores are mixed in.

### What we ablated

- **Pointer structure:** linear, MLP, and zero-initialized residual heads; the
  residual head won the controlled screen on development NLL and calibration.
- **Representation readout:** decision-token, query-mean, option-mean, and
  combined variants. Query-mean led the early screen; the release recipe uses
  decision-marker / option-close after the full model-family run.
- **Loss:** cross-entropy, pure InfoNCE, and mixed objectives; CE gave the best
  transfer accuracy in the loss sweep, while small contrastive terms mainly
  improved calibration. The released checkpoints use CE.
- **Readout family:** at 4B, direct-token improves JevBench (80.95% vs 80.09%),
  while pointer is slightly stronger on Transfer-v9 (78.68% vs 78.20%).

### Reproducibility

The five Transfer-v9 reports cover 1,264/1,264 requests; the headline aggregate
is the fixed 1,046-item clean, knowable subset. JevBench uses the frozen
`jevbench-public-v1.4.2.2` conversion with 231 items. Exact suite hashes and
unrounded headline metrics are recorded in the machine-readable results. Release
manifests, checkpoint-native reload reports, and GPU loader/reconstruction
parity were checked before publishing.

[Machine-readable release results](results/model-family-v2.json) ·
[Evaluation protocols and historical results](docs/EVALUATION.md).

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
