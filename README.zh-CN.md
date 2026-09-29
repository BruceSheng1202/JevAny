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

JevAny 用于训练和部署 Jev 风格的决策模型：你可以微调开源语言模型，也可以使用预训练 checkpoint。两者共用兼容 Jev 格式的 [Python 和 HTTP API](docs/API.md)，输入状态、问题与候选答案，即可获得选择及各选项的概率。

<p align="center">
  <img src="docs/hero.png" alt="JevAny 训练与部署流程：多模态数据、RLCR/SFT 训练、统一 API、测试环境与应用示例" width="100%">
</p>

| 从这里开始 | JevAny 提供什么 |
|---|---|
| **[训练](#训练)** | 训练数据和统一的 SFT/RLCR 训练框架 |
| **[推理与部署](#推理与部署)** | 预训练模型、统一 API、测试环境与应用示例 |

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

## 训练

### 训练数据

训练数据沿用推理时的 `state` 和 `questions`，并为每个问题增加 `label`。
当前模型系列使用 1,772,725 条文本记录、共 2,180,242 个有标签决策训练。
公开信息仅包含总规模与大类：偏好、Agent/工具决策、推理、分类和安全；
不公开数据混合比例和具体来源明细。

| 数据 | 提供的内容 | 使用入口 |
|---|---|---|
| 随包入门数据 | 用于熟悉训练流程的小型合成数据集 | `jevany data init --out data/starter` |
| 公开数据构建器 | 文本、图片和视频决策数据 | [数据构建指南](docs/TRAINING.md#data-beyond-the-starter) |
| 自己的数据 | 按统一 JSONL 格式添加标签的请求 | [格式与示例](docs/DATA.md) |

训练前先准备并校验入门数据：

```bash
jevany data init --out data/starter
jevany data validate data/starter/train.jsonl
```

### SFT

监督微调让 Jev 模型学习带标签的决策。在 CUDA GPU 上运行入门 recipe：

```bash
jevany train --config recipes/sft.toml --dry-run
jevany train --config recipes/sft.toml
```

Checkpoint 保存到 `runs/my-jev`。使用自己的数据时，添加 `--data data/my-domain.jsonl --out runs/domain-jev`。微调已发布的 Jev 模型可使用 [`recipes/finetune.toml`](recipes/finetune.toml)。

### RLCR

RLCR（Reinforcement Learning with Calibration Rewards）在 SFT 后继续训练，奖励同时考虑答案是否正确及其置信度。完成上面的 SFT recipe 后，运行：

```bash
jevany train --config recipes/rlcr.toml
```

该 recipe 从 `runs/my-jev` 继续训练，保存到 `runs/my-jev-rlcr`。RLCR 仍在开发完善中，详见[训练目标](docs/ALGORITHM.md#rlcr)。

支持的基座、图片和视频能力，以及本地 GPU 用法见[训练指南](docs/TRAINING.md#backbone-support)。

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

两种方式都只做一次 backbone prefill，不生成答案文本，因此同等输入下推理
速度应基本相当。Pointer 使用一个小型可学习 head 对决策标记和选项表示打分；
它支持超过 255 个选项，实际边界由上下文窗口决定。Direct-token 不使用专用
决策 head，而是给每个选项加上 255 个固定单 token 标签之一，再通过基座原始
LM head 打分。已发布 direct-token 模型在 4B JevBench 上效果更好，但训练时
使用 full-vocabulary cross-entropy，因此训练更慢，并且最多支持 255 个选项。

### 发布计划

当前版本发布下列五个 LoRA checkpoint。Full-parameter SFT、进一步优化的
post-training，以及超参优化版本计划后续发布，不包含在本次版本中。

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

统一报告两个固定协议的准确率：Transfer-v9 的 1,046 个 clean、knowable
决策，以及 JevBench 全部 231 个公开 development 项。所有本地评测均完整
覆盖，没有拒绝或截断样本。JevBench public 是开发集诊断结果，不是
Benchmark Heaven 的 sealed 官方分数。

| 发布模型 | Readout | Transfer-v9 | JevBench 准确率 | NLL ↓ | Brier ↓ | ECE ↓ |
|---|---|---:|---:|---:|---:|---:|
| Gemma 4B LoRA | Pointer | 70.84% | 77.49% | 0.536 | 0.309 | 0.043 |
| Qwen3.5 4B LoRA | Pointer | **78.68%** | 80.09% | 0.455 | 0.259 | 0.037 |
| Qwen3.5 4B Direct-Token LoRA | Direct-token | 78.20% | **80.95%** | 0.433 | 0.256 | 0.051 |
| Qwen3.8 27B LoRA | Pointer | **85.76%** | **90.48%** | **0.270** | **0.145** | 0.036 |
| Muse Glimmer 30B LoRA | Pointer | 83.46% | 87.45% | 0.316 | 0.174 | **0.027** |

同一组公开协议下的参考模型结果如下：

| 参考模型 | Transfer-v9 | JevBench public | 来源 |
|---|---:|---:|---|
| Kev-4B | 74.19% | 75.32% | 本地公开 checkpoint 评测 |
| Jev 1.13.0 | 85.37% | 86.58% | 本地 API 评测 / 已发布 JevBench 参考值 |
| Laya (`55cf4c4`) | 52.29% | 58.01% | 本地固定 revision 评测 |

Direct-token 4B 在已发布 4B 模型中取得更高的 JevBench 准确率，Pointer
4B 则在 Transfer-v9 上略高。参考模型之间只比较准确率；校准指标仅列出由
同一本地协议得到的发布模型结果，不混用不可用的 sealed 测试分数。

[本次发布的机器可读结果](results/model-family-v2.json) ·
[评测协议与历史结果](docs/EVALUATION.md)。

## Supported Model Families

![支持的 26 个模型，涵盖 Qwen、Gemma、Muse、Mistral、GLM、Nemotron 和 Llama](docs/supported-model-families.svg)

[模型 ID、支持的输入与运行要求](docs/TRAINING.md#backbone-support)。

## 文档与贡献

[训练](docs/TRAINING.md) · [部署](docs/DEPLOYMENT.md) · [API 兼容性](docs/API.md) · [数据](docs/DATA.md) · [评测](docs/EVALUATION.md) · [贡献指南](CONTRIBUTING.md)

JevAny 独立于 Jev 和 TypeSafe，不包含 Jev 权重或私有实现。部分基础设施改编自 [Kev](https://github.com/jaredpalmer/kev)，归属说明见 [NOTICE](NOTICE) 和 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)。代码和入门数据采用 Apache-2.0；基础模型与上游数据集保留各自条款。
