<p align="center">
  <img src="docs/title.png" alt="JevAny：从任意模型构建你的 Jev，部署到任意应用" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/collections/tianxinwei/jevany-adaptive-decision-systems-6ab2c941bcecb4d2c61d1326"><img alt="模型" src="https://img.shields.io/badge/%F0%9F%A4%97-checkpoints-ffb000"></a>
  <a href="docs/API.md"><img alt="API 文档" src="https://img.shields.io/badge/docs-API-0ea5e9"></a>
  <a href="docs/CASES.md"><img alt="示例" src="https://img.shields.io/badge/examples-gallery-8b5cf6"></a>
  <a href="pyproject.toml"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="https://github.com/SimpleJev/JevAny/actions"><img alt="测试" src="https://github.com/SimpleJev/JevAny/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="许可证" src="https://img.shields.io/badge/license-Apache--2.0-32d6c5"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong><br>
  <a href="#快速上手">⚡ 快速上手</a> ·
  <a href="#演示">🎮 演示</a> ·
  <a href="#预训练模型">🤗 模型</a> ·
  <a href="#评测">📊 评测</a> ·
  <a href="#训练">🛠️ 训练</a> ·
  <a href="#文档与贡献">📚 文档</a>
</p>

**JevAny 是面向决策模型训练与部署的开源 infra。** 你可以直接使用已发布模型，也可以用自己的数据训练，
用于工单分流、工具选择和机器人动作决策。统一 API 接收状态、问题和候选选项，
直接返回选择结果与各选项概率，无需生成答案文本。

<p align="center">
  <img src="docs/hero.png" alt="JevAny infra：决策模型训练、部署与应用集成" width="100%">
</p>

## 演示

以下 30 个精选成功案例展示了 JevAny 在机器人、浏览器、软件、实验室和出行任务中的动作选择。
演示使用的模型是 `JevAny-27B-SFT`。

[![JevAny 在机器人、浏览器、软件、实验室和出行任务中选择动作](docs/demos/jevany-cases.gif)](docs/CASES.md)

[查看全部案例](docs/CASES.md)，或按下面的步骤打开演示，查看记录的动作和选项概率。

## 快速上手

### 打开交互演示

使用 Python 3.12 或更新版本，克隆仓库并安装轻量客户端：

```bash
git clone https://github.com/SimpleJev/JevAny.git
cd JevAny
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
jevany demo
```

打开 `http://127.0.0.1:8090`，点击 **Replay** 观看录制的运行过程。
只需 CPU 即可播放内置录制内容。在终端按 Ctrl+C 停止演示。

以下命令均在仓库根目录运行，并使用上述虚拟环境。

### 给客服工单选择处理部门

安装推理依赖，在 CUDA GPU 上启动已发布的 Qwen 4B 模型。
显存要求见[硬件与加载说明](docs/DEPLOYMENT.md#checkpoints-and-hardware)。

```bash
python -m pip install -e '.[serve,multimodal]'
jevany serve --checkpoint tianxinwei/JevAny-Qwen3.5-4B-LoRA \
  --device cuda --dtype bf16 --port 8008
```

保持服务运行，在使用相同虚拟环境的 Python 会话中，发送工单和候选处理部门：

```python
from jevany import Choice, JevClient

jev = JevClient("http://127.0.0.1:8008")
result = jev.system_one(
    state={"ticket": "I was charged twice. Please help."},
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Payment problems", "shipping": "Delivery problems"},
        ),
    },
)
answer = result["answers"]["department"]
print("Selected team:", answer["choice"])
print("Probabilities:", answer["probabilities"])
```

`choice` 返回一个候选部门名称，`probabilities` 返回各部门的概率。
你可以据此分配工单，也可以在结果不确定时转交人工审核。

二分类问题使用 `Noul`，例如判断工单是否需要紧急处理；有序评分使用 `Score`，
例如低、普通、高三个优先级。三类问题的完整格式见 [API 文档](docs/API.md)。
进程内推理可以[在 Python 中加载模型](docs/DEPLOYMENT.md#python)，通过相同接口调用。
图片和视频输入见[媒体配置](docs/DEPLOYMENT.md#native-media-and-limits)。

## 示例与测试环境

交互演示包含以下三个环境。动图展示 `JevAny-27B-SFT` 的加速回放，以及记录的动作和选项概率。

### [机械臂插孔](examples/README.md#robot-peg-insertion)

控制 Franka 夹爪抓取、对准并插入工件，由 PyBullet 接触物理验证结果。

![机械臂浏览器回放：Franka 插孔动作、模型原始选项概率和物理成功检查](docs/demos/playground-arm.gif)

### [Doom 走廊 · 3D](examples/README.md#doom-corridor-3d)

击败最后一个房间中左右两侧的敌人，再向前移动。使用 ViZDoom 和随包提供的 Freedoom 资源。

![Doom checkpoint 回放：击杀左右两名敌人后继续前进](docs/demos/playground-doom.gif)

### [Crafter 生存建造 · 2D](examples/README.md#crafter-survival-2d)

采集木材、制作工具、开采石头，同时管理生命值和物资。

![Crafter 浏览器回放：资源采集、制作工具和四项目标的完成进度](docs/demos/playground-crafter.gif)

### 让模型在环境中运行

停止只播放回放的演示，保持模型服务运行。安装可选游戏引擎，再连接模型服务启动演示：

```bash
python -m pip install -e '.[demo]'
jevany demo --base-url http://127.0.0.1:8008 --text-only
```

在浏览器中选择 **Run model** 让模型操作，或选择 **Play yourself** 自己操作。
实时控制向模型发送文本状态；机械臂控制使用 `.[robotics]` 依赖。
平台要求与环境接口见[演示指南](examples/README.md)，结合 LLM 规划器使用
JevAny 决策可参考[集成文档](docs/INTEGRATIONS.md)。

## 预训练模型

| 模型 | Readout | 用途 |
|---|---|---|
| <img src="docs/model-logos/jevany-gemma.svg" width="24" height="24" align="middle" alt="">&nbsp;[JevAny-Gemma-4B-LoRA](https://huggingface.co/tianxinwei/JevAny-Gemma-4B-LoRA) | Pointer | 轻量 Gemma 版本 |
| <img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;[JevAny-Qwen3.5-4B-LoRA](https://huggingface.co/tianxinwei/JevAny-Qwen3.5-4B-LoRA) | Pointer | 轻量、支持灵活选项数 |
| <img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;[JevAny-Qwen3.5-4B-Direct-Token-LoRA](https://huggingface.co/tianxinwei/JevAny-Qwen3.5-4B-Direct-Token-LoRA) | Direct-token | 当前 4B JevBench 最优版本 |
| <img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;[JevAny-Qwen3.8-27B-LoRA](https://huggingface.co/tianxinwei/JevAny-Qwen3.8-27B-LoRA) | Pointer | 默认模型；当前发布准确率最高 |
| <img src="docs/model-logos/jevany-muse.svg" width="24" height="24" align="middle" alt="">&nbsp;[JevAny-Muse-Glimmer-30B-LoRA](https://huggingface.co/tianxinwei/JevAny-Muse-Glimmer-30B-LoRA) | Pointer | Muse Glimmer 版本 |

这些仓库发布的是 LoRA adapter，加载时还需要对应基座，并适用基座模型的
许可证和访问条款。BF16 基座权重大约需要参数量两倍的字节数，另需运行时
显存。详见[硬件与加载说明](docs/DEPLOYMENT.md#checkpoints-and-hardware)。

Pointer 和 direct-token 模型使用相同 API。Pointer 在上下文允许的范围内支持最多
4,096 个选项，direct-token 最多支持 255 个。
训练与准确率的取舍见[输出方式说明](docs/TRAINING.md#pointer-and-direct-token-readouts)。

## 评测

Qwen3.8 27B 在两项评测中准确率最高，NLL 和 Brier 也最低。
4B 版本中，direct-token 的 JevBench 准确率最高，Pointer 的 Kev Transfer-v9 准确率最高。

<div align="center">

| 模型 | Kev Transfer-v9 ↑ | JevBench ↑ | NLL ↓ | Brier ↓ | ECE ↓ |
|:---|---:|---:|---:|---:|---:|
| <img src="docs/model-logos/kev.svg" width="24" height="24" align="middle" alt="">&nbsp;Kev-4B | 74.19% | 75.32% | 0.858 | 0.380 | 0.125 |
| <img src="docs/model-logos/kev.svg" width="24" height="24" align="middle" alt="">&nbsp;Kev-27B | 82.31% | 85.28% | 0.533 | 0.265 | 0.050 |
| <img src="docs/model-logos/typesafe.png" width="24" height="24" align="middle" alt="">&nbsp;Jev 1.13.0 | 85.37% | 86.58% | 0.644 | 0.212 | 0.033 |
| <img src="docs/model-logos/laya.svg" width="24" height="24" align="middle" alt="">&nbsp;Laya | 52.29% | 58.01% | 1.264 | 0.615 | 0.127 |
| **JevAny Releases** |  |  |  |  |  |
| <img src="docs/model-logos/jevany-gemma.svg" width="24" height="24" align="middle" alt="">&nbsp;Gemma 4B LoRA | 70.84% | 77.49% | 0.706 | 0.369 | 0.056 |
| <img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;Qwen3.5 4B LoRA | 78.68% | 80.09% | 0.587 | 0.297 | 0.035 |
| <img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;Qwen3.5 4B Direct-Token LoRA | 78.20% | 80.95% | 0.564 | 0.291 | **0.029** |
| <img src="docs/model-logos/jevany-muse.svg" width="24" height="24" align="middle" alt="">&nbsp;Muse Glimmer 30B LoRA | 83.46% | 87.45% | 0.464 | 0.229 | 0.032 |
| <img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;**Qwen3.8 27B LoRA** | **85.76%** | **90.48%** | **0.392** | **0.200** | 0.030 |

NLL、Brier 和 ECE 均在 Kev Transfer-v9 上计算。

</div>

[完整结果与评测协议](docs/EVALUATION.md#model-family-v2) ·
[机器可读结果](results/model-family-v2.json) ·
[方法与消融实验报告](docs/JEVANY_METHOD_AND_ABLATIONS.pdf)

## 训练

训练数据沿用推理时的 `state` 和 `questions`，为每个问题增加标签。
先用随包提供的合成客服工单开始训练，再换成自己的标注数据。

入门配置在 CUDA 上以 BF16 训练 Qwen3.5-0.8B，结果写入 `runs/my-jev`：

```bash
python -m pip install -e '.[train]'
jevany data init --out data/starter
jevany data validate data/starter/train.jsonl
jevany train --config recipes/sft.toml --dry-run
jevany train --config recipes/sft.toml
```

训练完成后，用随包提供的工单请求试用模型：

```bash
jevany decide examples/request.json --checkpoint runs/my-jev
```

通过 `--data` 指定自己的 JSONL，或用
[`recipes/finetune.toml`](recipes/finetune.toml) 微调已发布的 27B 模型。
CPU 配置、多模态数据和标准 `torchrun` 启动方式见[训练指南](docs/TRAINING.md)。
训练图片或视频模型时，安装 `.[train,multimodal]`。

### RLCR（实验功能）

完成 SFT 后，可以继续使用兼顾正确率与概率校准的奖励训练：

```bash
jevany train --config recipes/rlcr.toml
```

当前发布系列采用 LoRA SFT，训练数据包含 1,772,725 条文本记录和 2,180,242 个有标签决策。
全参数 SFT 与进一步的后训练改进仍在计划中。

[数据格式](docs/DATA.md) · [RLCR 目标](docs/ALGORITHM.md#rlcr) ·
[训练算力与实验说明](docs/JEVANY_METHOD_AND_ABLATIONS.pdf)

## 支持的模型系列

![支持的 26 个模型，涵盖 Qwen、Gemma、Muse、Mistral、GLM、Nemotron 和 Llama](docs/supported-model-families.svg)

[模型 ID、支持的输入与运行要求](docs/TRAINING.md#backbone-support)。

## 文档与贡献

[训练](docs/TRAINING.md) · [部署](docs/DEPLOYMENT.md) · [API](docs/API.md) · [数据](docs/DATA.md) · [评测](docs/EVALUATION.md) · [贡献指南](CONTRIBUTING.md)

欢迎贡献模型适配、评测或应用示例，开发步骤见[贡献指南](CONTRIBUTING.md)。
[方法报告](docs/JEVANY_METHOD_AND_ABLATIONS.pdf)及其
[LaTeX 源文件](docs/JEVANY_METHOD_AND_ABLATIONS.tex)介绍了模型设计与实验。

代码和入门数据采用 Apache-2.0。部分组件改编自 [Kev](https://github.com/jaredpalmer/kev)，归属说明见 [NOTICE](NOTICE) 和 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)。基座模型与上游数据集保留各自条款。
