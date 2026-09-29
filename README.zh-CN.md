<p align="center">
  <img src="docs/title.png" alt="JevAny：从任意模型构建你的 Jev，部署到任意应用" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/collections/tianxinwei/jevany-adaptive-decision-systems-6ab2c941bcecb4d2c61d1326"><img alt="模型" src="https://img.shields.io/badge/%F0%9F%A4%97-checkpoints-ffb000"></a>
  <a href="docs/API.md"><img alt="API 文档" src="https://img.shields.io/badge/docs-API-0ea5e9"></a>
  <a href="docs/CASES.md"><img alt="示例" src="https://img.shields.io/badge/examples-gallery-8b5cf6"></a>
  <a href="pyproject.toml"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="https://github.com/weitianxin/JevAny/actions"><img alt="测试" src="https://github.com/weitianxin/JevAny/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="许可证" src="https://img.shields.io/badge/license-Apache--2.0-32d6c5"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <strong>一个决策 API，五个开放 LoRA 模型，覆盖任意有限选项决策。</strong><br>
  输入状态、问题与候选答案，直接返回校准后的选择概率，无需生成答案文本。
</p>

<p align="center">
  <img src="docs/hero.png" alt="JevAny 训练与部署流程：多模态数据、RLCR/SFT 训练、统一 API、测试环境与应用示例" width="100%">
</p>

| 🤗 [模型](#预训练模型) | 📊 [结果](#评测) | ⚡ [部署](#推理与部署) | 🛠️ [训练](#训练) |
|---|---|---|---|
| Gemma、Qwen、Muse | Transfer + JevBench | Python + HTTP | SFT + RLCR |

## 演示

以下案例由较早但接口兼容的 JevAny checkpoint 录制：

[![JevAny 在机器人、浏览器、软件、实验室和出行任务中选择动作](docs/demos/jevany-cases.gif)](docs/CASES.md)

[查看 30 个精选成功案例](docs/CASES.md)，或通过[示例与测试环境](#示例与测试环境)试用自己的模型。

## 安装

使用 Python 3.12 或更新版本。克隆仓库并创建环境：

```bash
git clone https://github.com/weitianxin/JevAny.git
cd JevAny
python3.12 -m venv .venv
source .venv/bin/activate
```

按用途选择依赖：

| 用途 | 安装命令 |
|---|---|
| 调用已有 HTTP 服务 | `python -m pip install -e .` |
| 训练文本模型 | `python -m pip install -e '.[train]'` |
| 在本地运行文本模型或启动 HTTP 服务 | `python -m pip install -e '.[serve]'` |
| 运行支持原生媒体输入的已发布模型 | `python -m pip install -e '.[serve,multimodal]'` |

只安装客户端不会引入 PyTorch。训练图片/视频模型时，使用 `.[train,multimodal]`。以下命令均在仓库根目录运行；具体模型的硬件要求见[预训练模型](#预训练模型)。

## 预训练模型

| 模型 | Readout | 用途 |
|---|---|---|
| [JevAny-Gemma-4B-LoRA](https://huggingface.co/tianxinwei/JevAny-Gemma-4B-LoRA) | Pointer | 轻量 Gemma 版本 |
| [JevAny-Qwen3.5-4B-LoRA](https://huggingface.co/tianxinwei/JevAny-Qwen3.5-4B-LoRA) | Pointer | 轻量、支持灵活选项数 |
| [JevAny-Qwen3.5-4B-Direct-Token-LoRA](https://huggingface.co/tianxinwei/JevAny-Qwen3.5-4B-Direct-Token-LoRA) | Direct-token | 当前 4B JevBench 最优版本 |
| [JevAny-Qwen3.8-27B-LoRA](https://huggingface.co/tianxinwei/JevAny-Qwen3.8-27B-LoRA) | Pointer | 默认模型；当前发布准确率最高 |
| [JevAny-Muse-Glimmer-30B-LoRA](https://huggingface.co/tianxinwei/JevAny-Muse-Glimmer-30B-LoRA) | Pointer | Muse Glimmer 版本 |

这些仓库发布的是 LoRA adapter，加载时还需要对应基座，并适用基座模型的
许可证和访问条款。BF16 基座权重大约需要参数量两倍的字节数，另需运行时
显存。详见[硬件与加载说明](docs/DEPLOYMENT.md#checkpoints-and-hardware)。

### Pointer 与 direct-token

- **Pointer：**用轻量可学习 head 比较决策标记与选项表示；训练更高效，支持
  超过 255 个选项，实际边界由上下文窗口决定。
- **Direct-token：**用基座 LM head 对 255 个固定单 token 标签打分；4B
  JevBench 更高，但 full-vocabulary 训练更慢，最多支持 255 个选项。
- **推理：**两者都只做一次 backbone prefill、无需生成答案文本，同等输入下
  延迟应基本相当。

### 发布计划

- **当前：**五个已验证的 LoRA checkpoint。
- **下一步：**Full-parameter SFT。
- **后续：**改进的 post-training 与超参优化版本。

## 推理与部署

### Python API

[HTTP 服务](#http-服务)启动后，发送状态和带有候选选项的问题。返回值包含选中的选项及各选项的概率：

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

二分类问题使用 `Noul`，有序评分使用 `Score`。[API 文档](docs/API.md) 介绍了三种问题类型及兼容 Jev 的请求/返回格式。

### HTTP 服务

运行默认发布模型时，安装 `.[serve,multimodal]`，并使用满足 [27B 硬件要求](#预训练模型)的设备：

```bash
jevany serve --checkpoint tianxinwei/JevAny-Qwen3.8-27B-LoRA \
  --device cuda --dtype bf16 --port 8008
```

部署前面 SFT 示例训练的小模型时，运行：

```bash
jevany serve --checkpoint runs/my-jev --model-name my-jev --port 8008
```

### 进程内推理

在应用中加载一次 checkpoint，复用上例中的 `state` 和 `questions`：

```python
from jevany import JevModel

jev = JevModel.from_pretrained("runs/my-jev", model_name="my-jev")
result = jev.system_one(state=state, questions=questions)
```

更多模型调用方式见[部署指南](docs/DEPLOYMENT.md)，图片和视频输入见[媒体配置](docs/DEPLOYMENT.md#native-media-and-limits)。

## 示例与测试环境

在本地浏览器中打开交互演示。内置回放不需要 GPU、模型下载或推理服务：

```bash
python -m pip install -e .
jevany demo
```

以下动图来自较早但接口兼容的 JevAny checkpoint，保留了模型的实际选择和原始选项概率。

### [机械臂插孔](examples/README.md#robot-peg-insertion)

控制 Franka 夹爪抓取、对准并插入工件，由 PyBullet 接触物理验证结果。

![机械臂浏览器回放：Franka 插孔动作、模型原始选项概率和物理成功检查](docs/demos/playground-arm.gif)

### [Doom 走廊 · 3D](examples/README.md#doom-corridor-3d)

从最后一个房间开始，击杀左右两名敌人，再继续前进。使用 ViZDoom 和随包提供的 Freedoom 资源。

![Doom checkpoint 回放：击杀左右两名敌人后继续前进](docs/demos/playground-doom.gif)

### [Crafter 生存建造 · 2D](examples/README.md#crafter-survival-2d)

采集木材、制作工具、开采石头，同时管理生命值和物资。

![Crafter 浏览器回放：资源采集、制作工具和四项目标的完成进度](docs/demos/playground-crafter.gif)

### 实时控制

安装可选游戏引擎后，可以自己操作，也可以连接[已启动的模型服务](#http-服务)，在浏览器中选择 **Run model**：

```bash
python -m pip install -e '.[demo]'
jevany demo --base-url http://127.0.0.1:8008 --text-only
```

实时模型决策目前使用文本状态。机械臂控制使用单独的 `.[robotics]` 依赖。安装步骤、平台要求和环境接口见[演示指南](examples/README.md)，结合 LLM 规划器使用 Jev 决策可参考[集成文档](docs/INTEGRATIONS.md)。

## 评测

统一报告两个固定协议：Transfer-v9 的 1,046 个 clean、knowable 决策，以及
JevBench 全部 231 个公开 development 项。每次评测均完整覆盖全部样本。
JevBench public 是开发集诊断结果，不是 sealed 榜单分数。

| 模型 | 类型 | Transfer-v9 | JevBench | JB NLL ↓ | JB Brier ↓ | JB ECE ↓ |
|---|---|---:|---:|---:|---:|---:|
| Kev-4B | 参考模型 | 74.19% | 75.32% | — | — | — |
| Jev 1.13.0 | 参考模型 | 85.37% | 86.58% | — | — | — |
| Laya (`55cf4c4`) | 参考模型 | 52.29% | 58.01% | — | — | — |
| **JevAny Releases** |  |  |  |  |  |  |
| Gemma 4B LoRA | Pointer | 70.84% | 77.49% | 0.536 | 0.309 | 0.043 |
| Qwen3.5 4B LoRA | Pointer | 78.68% | 80.09% | 0.455 | 0.259 | 0.037 |
| Qwen3.5 4B Direct-Token LoRA | Direct-token | 78.20% | 80.95% | 0.433 | 0.256 | 0.051 |
| Muse Glimmer 30B LoRA | Pointer | 83.46% | 87.45% | 0.316 | 0.174 | **0.027** |
| **Qwen3.8 27B LoRA** | **Pointer** | **85.76%** | **90.48%** | **0.270** | **0.145** | 0.036 |

### Transfer-v9 分项

| 数据集 / 鲁棒性切片 | n | Gemma 4B | Qwen 4B P | Qwen 4B DT | Muse 30B | Qwen 27B |
|---|---:|---:|---:|---:|---:|---:|
| Emotion | 80 | 75.00% | 86.25% | 85.00% | 81.25% | 90.00% |
| PAWS | 80 | 80.00% | 75.00% | 77.50% | 77.50% | 86.25% |
| QNLI | 80 | 91.25% | 93.75% | 95.00% | 92.50% | 95.00% |
| TweetEval offensive | 80 | 80.00% | 81.25% | 81.25% | 83.75% | 83.75% |
| MMLU | 80 | 65.00% | 77.50% | 75.00% | 83.75% | 86.25% |
| MMLU-Pro | 200 | 38.50% | 58.50% | 52.50% | 61.00% | 68.00% |
| SciQ | 80 | 97.50% | 97.50% | 98.75% | 98.75% | 97.50% |
| Buried instruction（Emotion/PAWS/QNLI/TweetEval） | 80 | 70.00% | 77.50% | 78.75% | 82.50% | 78.75% |
| Compositional holdouts（AND/OR/conditional） | 96 | 67.71% | 87.50% | 86.46% | 95.83% | 90.62% |
| Contrastive policy（authorization/deadline） | 80 | 77.50% | 76.25% | 83.75% | 98.75% | 97.50% |
| Knowable controls（11 类 policy 任务） | 110 | 81.82% | 81.82% | 81.82% | 90.91% | 92.73% |
| **总计** | **1,046** | **70.84%** | **78.68%** | **78.20%** | **83.46%** | **85.76%** |

Direct-token 4B 在已发布 4B 模型中取得更高的 JevBench 准确率，Pointer
4B 则在 Transfer-v9 上略高。参考模型之间只比较准确率；校准指标仅列出由
同一本地协议得到的发布模型结果，不混用不可用的 sealed 测试分数。
Kev 与 Laya 来自本地公开 checkpoint 评测（Laya 固定到 `55cf4c4`）；Jev
采用本地 API 的 Transfer-v9 结果与其公开发布的 JevBench 结果。

### Ablation 摘要

- **Pointer 结构：**比较 linear、MLP 和 zero-initialized residual head；
  residual 在受控实验中取得更好的 development NLL 与校准表现。
- **表示 readout：**比较 decision token、query mean、option mean 和 combined；
  query-mean 在早期筛选中领先，完整模型系列最终采用 decision-marker / option-close。
- **Loss：**比较 cross-entropy、纯 InfoNCE 和混合目标；CE 在 loss sweep 中
  Transfer 准确率最高，小权重 contrastive 项主要改善校准。本次发布使用 CE。
- **Readout family：**4B direct-token 的 JevBench 更高（80.95% vs 80.09%），
  pointer 的 Transfer-v9 略高（78.68% vs 78.20%）。

### 可复现性

五次 Transfer-v9 运行均覆盖 1,264/1,264 个请求，headline 使用固定的 1,046
个 clean、knowable 样本。JevBench 固定为 `jevbench-public-v1.4.2.2`，共 231
项；完整 suite hash 与未舍入 headline 指标记录在机器可读结果中。发布前已核对 release
manifest、checkpoint 自带 reload report，并完成 GPU loader/reconstruction
logits 等价测试。

[本次发布的机器可读结果](results/model-family-v2.json) ·
[评测协议与历史结果](docs/EVALUATION.md)。

## 训练

当前模型系列使用 **1,772,725 条文本记录 / 2,180,242 个有标签决策**训练，
覆盖偏好、Agent/工具决策、推理、分类和安全。

准备随包提供的入门数据：

```bash
jevany data init --out data/starter
jevany data validate data/starter/train.jsonl
```

### SFT

```bash
jevany train --config recipes/sft.toml --dry-run
jevany train --config recipes/sft.toml
```

通过 `--data` 使用自己的 JSONL，或用 [`recipes/finetune.toml`](recipes/finetune.toml)
继续微调已发布模型。

### RLCR

在 SFT checkpoint 上继续进行兼顾正确率与校准的奖励训练：

```bash
jevany train --config recipes/rlcr.toml
```

[训练指南](docs/TRAINING.md) · [数据格式](docs/DATA.md) ·
[RLCR 目标](docs/ALGORITHM.md#rlcr)

## Supported Model Families

![支持的 26 个模型，涵盖 Qwen、Gemma、Muse、Mistral、GLM、Nemotron 和 Llama](docs/supported-model-families.svg)

[模型 ID、支持的输入与运行要求](docs/TRAINING.md#backbone-support)。

## 文档与贡献

[训练](docs/TRAINING.md) · [部署](docs/DEPLOYMENT.md) · [API 兼容性](docs/API.md) · [数据](docs/DATA.md) · [评测](docs/EVALUATION.md) · [贡献指南](CONTRIBUTING.md)

JevAny 独立于 Jev 和 TypeSafe，不包含 Jev 权重或私有实现。部分基础设施改编自 [Kev](https://github.com/jaredpalmer/kev)，归属说明见 [NOTICE](NOTICE) 和 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)。代码和入门数据采用 Apache-2.0；基础模型与上游数据集保留各自条款。
