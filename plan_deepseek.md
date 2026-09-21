# G0.5 在 YAM 双臂上的复现计划（从零执行版·数据转换由你负责）

> **前提**：仓库是 `OpenGalaxea/GalaxeaVLA` 的 main 分支，没有任何新增文件。  
> **已从采集数据确认**：YAM 与官方 R1Lite **形状完全相同**（固定底座、双 6 轴臂、双夹爪），原始动作空间都是 **14 维**，经官方 merger 分组 pad 后都是 **27 维**。  
> **直接结论**：**不新建 parts_meta、不注册新 embodiment**，数据/任务配置里 embodiment key 保持 `galaxea_r1lite`，parts_meta 一律引用 `configs/data/parts_meta/r1lite.yaml`。  
> **目标**：在给定**任务指令文本**和当前观测的条件下，让微调后的 G0.5 输出 YAM 双臂的关节动作序列，完成对应抓放任务。  
> **任务指令文本**：**框架尚未确定**，但已知训练和部署都需要它作为模型输入。本计划保留这一步骤，具体格式、语言、映射方式后续再定。  
> **数据分工**：**采集由数据方负责；mcap → LeRobot 转换由你自己负责。**  
> **路线**：微调官方 ActionCodec + 微调官方 G0.5 backbone；不做 CoT。  
> **你的环境**：Windows 本机做代码准备和数据转换；Linux GPU 训练机做微调；YAM 工控机做真机部署。

---

## 第 0 步：先搞清楚整件事在干什么

### 0.1 目标和路线

**目的**：让预训练好的 G0.5 能控制你的 YAM 双臂机器人。

做法是：

1. **微调 ActionCodec**：ActionCodec 是 G0.5 的"动作翻译器"，把连续关节序列翻译成离散 token。官方 codec 是按 R1Lite 训练的，YAM 虽然结构相同，但关节限位、单位、动作分布不同，需要微调。
2. **微调 G0.5 backbone**：让模型在 YAM 的视觉、本体状态、任务指令文本、动作数据上适配。

**不做 CoT**：训练和推理都用 no-CoT 模板，模型不生成推理 token。论文本身在主要评估中也使用 no-CoT 格式。

**通过标准**：你能说清"为什么需要微调 codec 和 backbone"，以及"为什么不做 CoT"。

### 0.2 模型输入输出

**输入**：

| 输入 | 来源 | 说明 |
|---|---|---|
| 多路相机图像 | 机器人相机实时采集 | 训练时来自数据集；部署时来自工控机 |
| 本体状态（14 维） | 机器人关节编码器 | 训练时来自数据集；部署时来自工控机 |
| **任务指令文本** | **训练：数据集里每条 episode 附带；部署：客户端指定** | **格式待定**，但必须存在 |
| embodiment 标识 | 配置固定为 `galaxea_r1lite` | 由配置文件给出，不是模型生成 |

**输出**：动作 token → 解码成 14 维关节目标（弧度制）。

**关键澄清**：任务指令文本不是模型生成的，它是 G0.5 作为 VLA 的条件输入之一。训练数据里每条 episode 都挂一条指令，部署时客户端每次请求都带一条指令。**具体用什么文本、什么语言、什么映射方式，待你们确定。**

### 0.3 任务指令文本待定，但需保留此步骤

**目的**：明确任务指令文本是模型输入的一部分，虽然框架未定，但不能缺。

**当前状态**：任务指令框架尚未确定。你们需要知道：

- 训练数据中，每条 episode 必须有一个 `task` 字段，内容是一条字符串；
- 部署时，客户端发送的请求中必须包含 `task` 字段，内容是一条字符串；
- 训练和部署的指令文本应保持同一套格式，避免分布偏移；
- 具体是英文、中文、固定模板、动态生成，还是从任务 ID 映射，后续再定。

**本计划保留此步骤，不预设具体框架。** 你自己在转换数据时，可以先用一个占位字符串（例如任务 ID 或统一占位符）填入 `task` 字段，等框架确定后再重新转换或做后处理。

**通过标准**：你能说出"训练时 task 字段从哪来、部署时 task 字段从哪来、两者必须一致"。

### 0.4 YAM 直接复用 R1Lite 配置（已从数据确认）

**目的**：避免在代码里注册新本体，直接复用官方已有的 `galaxea_r1lite`。

**已从采集数据确认的事实**：

- YAM 和 R1Lite 都是固定底座、双 6 轴臂、双夹爪；
- 原始动作空间都是 14 维：`left[6 关节 + 1 夹爪] ++ right[6 + 1]`；
- 经官方 merger 分组 pad 后都是 27 维：
  ```
  left_control(9) | left_gripper(1) | right_control(9) | right_gripper(1) | lower_body(7 个零填充)
  ```

**直接结论**：

- **不新建 parts_meta**，一律引用 `configs/data/parts_meta/r1lite.yaml`；
- **不注册新 embodiment**，数据/任务配置里 embodiment key 保持 `galaxea_r1lite`；
- **不新建 `configs/data/parts_meta/yam.yaml`**；
- **不新建 `configs/tokenizer/actioncodec_yam.yaml`**；
- YAM 与 R1Lite 的真正差异（关节限位、相机位姿、通信协议）**只在部署客户端处理**。

**通过标准**：`configs/data/yam.yaml` 和 `configs/task/yam.yaml` 里都没有新增 YAM 专属的 parts_meta，embodiment key 保持 `galaxea_r1lite`。

### 0.5 数据分工

| 环节 | 谁负责 |
|---|---|
| 遥操作采集 | 数据方（实验室工控机） |
| 原始 mcap 上传/汇聚 | 数据方 |
| **mcap → LeRobot 转换** | **你（自己在训练机或 Windows 上跑）** |
| normalizer stats | 你（训练机上） |
| codec 微调 | 你（训练机上） |
| backbone 微调 | 你（训练机上） |
| 真机部署 | 你（工控机上） |

### 0.6 全流程

```
① 数据方采集并汇聚 YAM mcap 原始数据
② 你自己写 mcap_to_lerobot.py 并运行转换，输出到 data/yam/
③ 训练机上算 normalizer stats（只用 YAM 数据）
④ 训练机上微调 ActionCodec（用官方 codec 初始化）
⑤ 训练机上微调 G0.5 backbone（用官方 backbone 初始化）
⑥ 部署：服务端 + YAM 客户端
```

**注意**：官方权重只用于第 ④⑤ 步的初始化，跟第 ③ 步算 stats 无关。

---

## 第 1 步：认识仓库结构，找到参考文件

**目的**：知道 G0.5 的代码结构、配置体系、关键接口在哪里，为后面写配置和脚本做准备。

### 1.1 克隆仓库

```bash
git clone https://github.com/OpenGalaxea/GalaxeaVLA
cd GalaxeaVLA
```

### 1.2 仓库顶层结构

| 目录/文件 | 作用 |
|---|---|
| `configs/` | Hydra 配置中心，所有任务、数据、模型、分词器配置都在这里 |
| `scripts/` | 训练、部署、数据统计等入口脚本 |
| `src/g05/` | 核心代码：模型、ActionCodec、数据管线、处理器 |
| `tools/` | 辅助工具（配置解析、可视化等） |
| `experiments/` | 部署客户端示例（如 `r1lite/`） |
| `tests/` | 测试脚本和示例（如 `test_dataloader_batch.py`） |
| `QUICK_START.md` | 官方快速开始文档，优先阅读 |

### 1.3 重点掌握的内容

#### 1.3.1 配置体系（Hydra）

- 入口：`scripts/run/finetune.sh`，通过 `configs/task/<name>.yaml` 组合模型、分词器和数据配置。
- 关键配置目录：
  - `configs/task/`：任务级配置，如 `r1lite.yaml`，指定超参、数据入口、codec 路径。
  - `configs/data/`：数据级配置，如 `r1lite.yaml`，定义数据集路径、相机尺寸、动作分组、归一化方式。
  - `configs/model/`：模型架构配置。
  - `configs/tokenizer/`：ActionCodec 配置。
- 本实验只需新增 `configs/task/yam.yaml` 和 `configs/data/yam.yaml`，**不修改官方任何已有文件**。

#### 1.3.2 ActionCodec 与 wrapper 接口

- 核心文件：`src/g05/tokenizer/models/actioncodec2_v2/wrapper.py`
- 你需要掌握 `ActionCodecV2Wrapper` 的：
  - `__init__` 如何接收 `vq_config` 并加载权重；
  - `encode` / `forward` 的输入输出格式；
  - 内部属性：`key_dims`、`_nn_keys`、`_rule_keys`、`_model_arch_cfg`；
  - 冻结策略：如何冻结 `conv_in`、`encoder`、第 0 层主码本（EMA buffer）。
- 官方 **ActionCodec 训练脚本未开源**（内部名 `train_vq.py`），你需要自己写 `scripts/train_tokenizer.py`。

#### 1.3.3 数据格式（LeRobot v3.0）

- 训练数据必须是 LeRobot v3.0 格式。
- 参考 `configs/data/r1lite.yaml` 中的 `shape_meta`：
  - 动作/状态原始 14 维，布局为 `left[6 关节 + 1 夹爪] ++ right[6 + 1]`；
  - 相机键名：`observation.images.head_rgb`、`left_wrist_rgb`、`right_wrist_rgb`；
  - 动作经官方 merger 分组 pad 后为 27 维。
- **已从采集数据确认**：YAM 与 R1Lite 形状完全相同，因此直接复用 `configs/data/parts_meta/r1lite.yaml`，**不新建 parts_meta，不注册新 embodiment**。
- **任务指令文本**：每条 episode 需要一个 `task` 字段（字符串），具体内容待定。

#### 1.3.4 部署流程

- 参考 `experiments/r1lite/`，它是纯通信节点（numpy + websockets + msgpack，无 torch）。
- 本实验需新建 `experiments/yam/`，将 `core/communication/ros2_bridge.py` 替换为 YAM 自己的 SDK/通信层，其余主循环、协议、msgpack 直接搬。
- 服务端使用 `scripts/serve_policy.py`，加载微调后的 backbone 权重。
- 客户端请求中必须包含 `task` 字段，内容是一条字符串。

#### 1.3.5 官方未开源或不能直接用于 YAM 的部分

- `train_vq.py`：ActionCodec 训练脚本，需自写 `scripts/train_tokenizer.py`。
- `GalaxeaLeRobotToolkit`：只认 R1 的 ROS2 CDR 消息，读不了 YAM 的 JSON schema mcap，需自写 `tools/yam/mcap_to_lerobot.py`。

### 1.4 本实验改动清单

| 类别 | 文件/目录 | 操作 | 核心内容 |
|---|---|---|---|
| **新增** | `configs/data/yam.yaml` | 复制 `r1lite.yaml` 并修改 3 处 | 1. `dataset_dirs` 指向 `data/yam/`<br>2. 相机 `raw_shape` 改为 `[3, 480, 640]`<br>3. `action_filter` 改为 `DummyActionFilter` |
| | `configs/task/yam.yaml` | 复制 `r1lite.yaml` 并修改 3 处 | 1. `override /data: yam`<br>2. `datastatics_path` 指向 `data/stats/yam_stats.json`<br>3. `ckpt_dir` 指向 `checkpoints/action_tokenizer_yam.pt` |
| | `tools/yam/mcap_to_lerobot.py` | **你写、你跑** | YAM mcap → LeRobot 转换脚本 |
| | `tools/yam/tasks_yam.json` | **可选/待定** | 任务 ID → 指令映射，等任务指令框架确定后再建 |
| | `scripts/train_tokenizer.py` | 你写、你跑 | ActionCodec 微调脚本，复用官方 `ActionCodecV2Wrapper` |
| | `experiments/yam/` | 新建（后期） | YAM 部署客户端，参考 `experiments/r1lite` |
| **修改** | **无** | **不修改任何官方文件** | 所有改动通过新增 YAML 覆盖或引用官方配置实现 |
| **不新建** | `configs/data/parts_meta/yam.yaml` | 不需要 | 直接复用 `r1lite.yaml` |
| | `configs/tokenizer/actioncodec_yam.yaml` | 不需要 | 参数内联在 `configs/task/yam.yaml` 中 |
| | 新 embodiment 注册代码 | 不需要 | embodiment key 保持 `galaxea_r1lite` |

### 1.5 现在（Windows 本机）可做的准备

1. **阅读官方文档与关键代码**：
   ```bash
   cat QUICK_START.md
   code src/g05/tokenizer/models/actioncodec2_v2/wrapper.py
   ```

2. **创建 YAM 配置骨架**：
   ```bash
   cp configs/data/r1lite.yaml configs/data/yam.yaml
   cp configs/task/r1lite.yaml configs/task/yam.yaml
   mkdir -p tools/yam
   ```
   按 1.4 表修改这两份 YAML。

3. **任务指令框架**：**先保留此步骤，不创建 `tasks_yam.json`**。等你们确定指令格式后，再决定是否需要一个映射文件。

4. **写 `tools/yam/mcap_to_lerobot.py`**：参考官方 `GalaxeaLeRobotToolkit` 的对齐逻辑，处理 YAM 的 JSON schema mcap，输出 LeRobot v3.0。

5. **搭建 `scripts/train_tokenizer.py` 骨架**。

### 1.6 验证配置解析

**目的**：确认你写的 YAM 配置能被 Hydra 正确合并，且与 r1lite 的差异只在预期 3 处。

```bash
python tools/resolve_config.py yam --key data.embodiment_datasets.galaxea_r1lite.shape_meta.images
python tools/resolve_config.py yam --key model.model_arch
python tools/resolve_config.py yam --diff r1lite --only-diff
```

**通过标准**：只显示相机尺寸、数据集路径、codec 路径三处差异；`action_dim` / `proprio_dim` 仍为 27，parts_meta 仍指向 `r1lite.yaml`。

---

## 第 2 步：新建 YAM 数据配置 `configs/data/yam.yaml`

**目的**：写一份"数据说明书"，告诉训练框架去哪个目录读 YAM 数据、相机多大、怎么处理动作。

**通过标准**：`configs/data/yam.yaml` 存在，且与 `configs/data/r1lite.yaml` 的差异只在 3 处（数据集路径、相机尺寸、action_filter）。

### 2.1 做法

```bash
cp configs/data/r1lite.yaml configs/data/yam.yaml
```

**只改 3 处**。

### 2.2 三处修改

**改动 1：数据集路径**

找到 `embodiment_datasets.galaxea_r1lite.dataset_groups`，把 `dataset_dirs` 改成：

```yaml
dataset_dirs:
  - data/yam/TASK-YAM-0001_lerobot
  - data/yam/TASK-YAM-0002_lerobot
  - data/yam/TASK-YAM-0004_lerobot
  - data/yam/TASK-YAM-0006_lerobot
  - data/yam/TASK-YAM-0007_lerobot
  - data/yam/TASK-YAM-0008_lerobot
  - data/yam/TASK-YAM-0010_lerobot
  # 其余任务转换完成后在此追加
```

**改动 2：相机原始尺寸**

YAM 相机是 640×480，R1Lite 是 1280×720。找到两处 `shape_meta.images`，把三个相机的 `raw_shape` 从 `[3, 720, 1280]` 改成 `[3, 480, 640]`。

**改动 3：动作过滤**

改成 Dummy：

```yaml
action_filter:
  _target_: g05.data_processor.transforms.action_filter.DummyActionFilter
```

### 2.3 不要改的地方（重要）

- `embodiment_type` 保持 `galaxea_r1lite`；
- **`parts_meta` 引用统一到 `configs/data/parts_meta/r1lite.yaml`**；
- `action_state_merger` 保持官方 `PaddingActionMerger`；
- `RelativeJointTransform` 保持不动；
- 语言相关字段保持与 r1lite 一致，除非你明确要改。

### 2.4 验证

**目的**：确认相机尺寸和 parts_meta 引用正确。

```bash
python tools/resolve_config.py yam --key data.embodiment_datasets.galaxea_r1lite.shape_meta.images
```

**通过标准**：`raw_shape` 是 `[3, 480, 640]`。

---

## 第 3 步：新建 YAM 任务配置 `configs/task/yam.yaml`

**目的**：指定用哪个数据配置、用哪份 stats、用哪个 codec 权重、用什么模型超参。

**通过标准**：`configs/task/yam.yaml` 存在，且与 `configs/task/r1lite.yaml` 的差异只在 3 处（数据入口、stats 路径、codec 权重指向）。

### 3.1 做法

```bash
cp configs/task/r1lite.yaml configs/task/yam.yaml
```

### 3.2 三处修改

**改动 1：数据入口**

在 `defaults` 列表里把 `override /data: r1lite` 改成：

```yaml
- override /data: yam
```

**改动 2：stats 路径**

```yaml
datastatics_path: data/stats/yam_stats.json
```

**改动 3：codec 权重指向**

在 `tokenizer.vq_config` 下增加：

```yaml
tokenizer:
  vq_config:
    ckpt_dir: checkpoints/action_tokenizer_yam.pt
```

做基线对照时把这一行注释掉，会回落到官方 `checkpoints/action_tokenizer.pt`。

### 3.3 不要改的地方（重要）

- `action_dim` / `proprio_dim` 保持 27；
- **`parts_meta` 保持 `configs/data/parts_meta/r1lite.yaml`**；
- `predict_cot` 保持 `false`；
- `discrete_action` / `continuous_action` 保持 `true` / `false`；
- 语言相关字段保持 r1lite 原样。

### 3.4 验证

**目的**：确认任务配置解析正确，27 维动作空间没变。

```bash
python tools/resolve_config.py yam --key model.model_arch
python tools/resolve_config.py yam --diff r1lite --only-diff
```

**通过标准**：`action_dim` / `proprio_dim` 是 27；parts_meta 指向 `r1lite.yaml`；diff 只显示 3 处预期差异。

---

## 第 4 步：运行 mcap → LeRobot 转换

**目的**：把 YAM 原始 mcap 数据转换成训练能直接读的 LeRobot v3.0 数据集。

**通过标准**：`data/yam/TASK-YAM-XXXX_lerobot/` 存在，读回后相机 shape 是 `(3, 480, 640)`、action 是 `(6,)`。

### 4.1 装依赖

Windows：

```powershell
python -m pip install mcap av numpy pillow
```

Linux 训练机上还要 `uv sync` 装 GalaxeaVLA 环境。

### 4.2 目录要求

`--input-root` 指向的目录下，每个任务长这样：

```
<input-root>/
  TASK-YAM-0001/
    mcap/
      success/
        episode_0001.mcap
```

只扫 `mcap/success/`，其他目录不处理。

### 4.3 路径怎么传

**`--input-root` 传 `TASK-YAM-XXXX` 的上一级目录**，不是任务目录本身，也不是 mcap 文件。

| mcap 实际位置 | `--input-root` 传 |
|---|---|
| `F:\TASK-YAM-0001\mcap\success\episode_0001.mcap` | `F:\` |
| `/data/yam/TASK-YAM-0001/mcap/success/episode_0001.mcap` | `/data/yam` |

报 `在 XXX 下没找到 **/mcap/success/*.mcap` 就是传错了。

### 4.4 预览（Windows 本机）

**目的**：不写盘，只解析、对齐、出预览图，确认数据解析正确。

```powershell
python tools/yam/mcap_to_lerobot.py --input-root F:\ --only-task TASK-YAM-0001 --max-episodes 1 --preview
```

出图在 `./preview_out/_preview/`，打开 PNG 确认三路相机正常。

**通过标准**：4 张 PNG 都能打开，三路相机画面清晰、时间对齐、无花屏。

**预览通过后，Windows 这边的验证就完成了。** 正式写盘需要 `g05` 包，只能在训练机上跑。

### 4.5 正式转换（训练机）

先建占位 `tasks_yam.json`：

```bash
mkdir -p tools/yam
cat > tools/yam/tasks_yam.json << 'EOF'
{"TASK-YAM-0001": "placeholder"}
EOF
```

单条：

```bash
python tools/yam/mcap_to_lerobot.py \
  --input-root /data/yam --output-root /data/yam \
  --tasks-json tools/yam/tasks_yam.json \
  --only-task TASK-YAM-0001 --max-episodes 1 --overwrite
```

全量（去掉 `--only-task` 和 `--max-episodes`）：

```bash
python tools/yam/mcap_to_lerobot.py \
  --input-root /data/yam --output-root /data/yam \
  --tasks-json tools/yam/tasks_yam.json
```

### 4.6 转换后校验

**目的**：确认写出的 LeRobot 数据集 shape 正确。

```bash
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('/data/yam/TASK-YAM-0001_lerobot')
print('images:', ds[0]['observation.images.head_rgb'].shape)
print('action:', ds[0]['action.left_arm'].shape)
print('task:', ds[0].get('task'))
"
```

**通过标准**：images 是 `(3, 480, 640)`，action 是 `(6,)`，task 是占位符或你填的指令。

### 4.7 参数速查

| 参数 | 说明 |
|---|---|
| `--input-root` | mcap 根目录（任务目录的上一级） |
| `--output-root` | LeRobot 输出根目录 |
| `--tasks-json` | task id → 指令映射（preview 可省） |
| `--only-task` | 只处理某个任务 |
| `--max-episodes` | 每任务最多转几条 |
| `--overwrite` | 覆盖已有输出 |
| `--preview` | 只解析出图，不写数据集 |

### 4.8 补充

- `task` 字段现在是占位符，任务指令框架确定后需重转；
- 相机 CHW 布局脚本已处理正确，不用改；
- `robot_type` 不影响训练，随便填。

---

## 第 5 步：训练机准备（两件事分开做）

**目的**：为第 6、7 步微调准备初始化权重和归一化基础。

第 5 步包含两件**互相独立**的事情：

- **5.A 下载官方权重**：为第 6、7 步微调做准备，跟算 stats 无关；
- **5.B 算 normalizer stats**：完全用你自己的 YAM 数据，不涉及官方权重。

### 5.0 环境准备

```bash
sudo apt-get update && sudo apt-get install -y git ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
cd GalaxeaVLA-main
uv sync
```

### 5.A 下载官方权重（供第 6、7 步使用）

**目的**：把 G0.5 官方发布的预训练 codec 和 backbone 下载到本地，作为后续微调的初始化。

从 Hugging Face 下载：

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download OpenGalaxea/G05 --repo-type model --local-dir checkpoints
```

确认：

```bash
ls checkpoints/action_tokenizer.pt    # 官方 codec，第 6 步微调初始化
ls checkpoints/g05_base*              # 官方 backbone，第 7 步微调初始化
```

**说明**：

- 这两个文件是 G0.5 官方发布的**预训练产物**，不是从你的数据里学出来的；
- 你的 YAM 数据是用来微调它们，让它们适配 YAM；
- **它们不参与第 5.B 步算 stats。**

**通过标准**：`checkpoints/action_tokenizer.pt` 和 `checkpoints/g05_base*` 都存在。

### 5.B 算 normalizer stats（只用你自己的 YAM 数据）

**目的**：算出 YAM 数据的均值、方差、分位数，供 codec 和 backbone 共享同一套归一化基础。

**输入**：`configs/data/yam.yaml` 指向的 `data/yam/TASK-YAM-XXXX_lerobot/`（你第 4 步转换出来的数据）。

**不涉及官方权重。**

```bash
export HF_DATASETS_CACHE=/data/hf_cache
export GALAXEA_FM_OUTPUT_DIR=/data/output/g05_yam
export GALAXEA_FM_DATASET_STATS_CACHE_DIR=/data/stats_cache

python tests/test_dataloader_batch.py \
  --mixture configs/data/yam.yaml \
  --stats data/stats/yam_stats.json \
  --stats-downsample-rate 1
```

**输出**：`data/stats/yam_stats.json`。

**通过标准**：

- `data/stats/yam_stats.json` 存在；
- 每个关节维 std 不为 0；
- 夹爪维呈双峰（0/1）。

**这份 stats 是第 6、7 步共用的归一化基础，换数据必须重算。**

---

## 第 6 步：新建并运行 ActionCodec 微调脚本

**目的**：让官方 codec 适配 YAM 的动作分布。

**通过标准**：同一批 held-out chunk 上，YAM codec 的逐键 RMSE 不高于官方、码本利用率不低于官方。

### 6.1 为什么需要新建

官方 ActionCodec 训练脚本未开源（内部名 `train_vq.py`）。你需要自己写 `scripts/train_tokenizer.py`。

### 6.2 设计原则

- **不重写模型/损失**：用官方 `ActionCodecV2Wrapper`；
- **数据管线复用官方**：`instantiate_dataset / build_processors`；
- **存盘格式对齐官方 `load_model`**；
- **单卡即可**。

### 6.3 脚本流程

1. Hydra compose `configs/task/yam.yaml`；
2. 构造 train/eval 数据集，加载 `data/stats/yam_stats.json`；
3. 用 `model.tokenizer.vq_config` 构造 `ActionCodecV2Wrapper`；
4. 从 `checkpoints/action_tokenizer.pt`（**第 5.A 步下载的官方权重**）加载初始化；
5. 冻结策略：
   - **模式 A（推荐）**：冻结 `conv_in`、`encoder`、第 0 层主码本（EMA buffer），只训残差码本 + decoder；
   - **模式 B（对照）**：全参数小 LR；
6. 训练若干 epoch；
7. 评估：逐键 RMSE、OR@8steps、码本利用率；
8. 存到 `checkpoints/action_tokenizer_yam.pt`。

### 6.4 关键坑

- **第 0 层码本是 EMA buffer**，冻结时除了 `requires_grad=False`，还要把 `_ema_update / _replace_dead_codes / _init_codebook` 置为 no-op；
- **r1lite 默认 `val_set_proportion=1e-4`**，脚本里要覆盖成 0.02 左右；
- **collate 必须用官方 `collate_fn_pad_sequences`**。

### 6.5 运行

**步骤 6.5.1：基线评估（先确认能加载官方 codec）**

```bash
python scripts/train_tokenizer.py --task yam --eval-only \
    --ckpt checkpoints/action_tokenizer.pt
```

**通过标准**：打印出 RMSE、码本利用率等指标。

**步骤 6.5.2：微调（推荐模式 A）**

```bash
python scripts/train_tokenizer.py --task yam \
    --init-ckpt checkpoints/action_tokenizer.pt \
    --out checkpoints/action_tokenizer_yam.pt \
    --freeze encoder_cb0 --epochs 2 \
    --micro-batch 64 --grad-accum 4 --lr 5e-5
```

**通过标准**：训练循环无报错，loss 稳定下降，`checkpoints/action_tokenizer_yam.pt` 生成。

**步骤 6.5.3：评估微调后**

```bash
python scripts/train_tokenizer.py --task yam --eval-only \
    --ckpt checkpoints/action_tokenizer_yam.pt
```

**通过标准**：微调后指标不差于官方基线。

### 6.6 验收标准

同一批 held-out chunk：

- 逐键 RMSE **不高于**官方；
- OR@8steps **不低于**官方；
- 死码率 **不升高**。

达不到就回退：注释掉 `configs/task/yam.yaml` 里的 `tokenizer.vq_config.ckpt_dir`。

---

## 第 7 步：微调 G0.5 backbone

**目的**：让 backbone 在 YAM 的视觉、本体状态、任务指令文本、动作数据上适配。

**通过标准**：4 个子步骤逐级通过；正式训练日志显示 `Loaded N/N`、`Missing (rand init): 0`，loss 稳定下降，输出目录完整。

### 7.1 配置解析

**目的**：确认 Hydra 能正确合并 YAM 配置。

```bash
bash scripts/run/finetune.sh 1 yam --dry-run --max_datasets 1
```

**通过标准**：打印解析后的配置，无 `KeyError` / `Missing mandatory`。

### 7.2 数据闭环（随机初始化）

**目的**：验证数据能加载、codec 能在线编码、模型能前向，先不加载预训练权重。

```bash
bash scripts/run/finetune.sh 1 yam --test --max_datasets 1 \
    model.pretrained_ckpt=null model.max_steps=10
```

**通过标准**：

- 打印 `pixel_values` shape，与 `[B, 3, 224, 224]` 一致；
- 打印 codec 加载信息（LOSSLESS 或类似）；
- 10 步 loss 有值。

### 7.3 overfit sanity

**目的**：只喂 5 个样本跑 500 步，如果管线全对，loss 应能压到接近 0。

```bash
bash scripts/run/finetune.sh 1 yam --overfit_batch 5 model.max_steps=500
```

**通过标准**：

- `action_loss` 从初值（约 8.0）压到很低（< 0.1）；
- 无 NaN。

### 7.4 正式训练

**目的**：用全部 YAM 数据微调 backbone。

```bash
bash scripts/run/finetune.sh 8 yam logger.mode=offline
```

**通过标准**：

- 加载日志：`Loaded N/N`、`Missing (rand init): 0`；
- loss 稳定下降；
- 输出目录生成完整。

### 7.5 输出

`$G05_OUTPUT_DIR/yam/<exp>/` 里包含：

- `checkpoints/`
- `dataset_stats.json`
- `action_tokenizer.pt`
- `.hydra/config.yaml`

**加载日志必须显示**：`Loaded N/N`、`Missing (rand init): 0`。

**不要动 run 目录里的 sidecar 文件**。

---

## 第 8 步：部署推理

**目的**：把微调后的模型部署到服务端，工控机客户端通过网络调用，控制 YAM 双臂。

**通过标准**：服务端启动成功；客户端连通；实机任务测试通过；语言跟随与泛化测试通过。

### 8.1 服务端（GPU 机）

```bash
PYTHONPATH="src:${PYTHONPATH:-}" python scripts/serve_policy.py \
  --ckpt_path $G05_OUTPUT_DIR/yam/<exp>/checkpoints/step_xxxxx/model_state_dict.pt \
  --host 0.0.0.0 --port 8080 \
  eval_embodiment=galaxea_r1lite \
  model.model_weights_to_bf16=true \
  model.model_arch.attn_implementation=sdpa \
  model.model_arch.discrete_action=true \
  model.model_arch.continuous_action=false
```

**通过标准**：服务端打印监听地址，无报错。

### 8.2 客户端（工控机，新建 `experiments/yam/`）

照 `experiments/r1lite/` 裁剪：

- 替换通信层为 YAM SDK；
- 上行：三路图像、14 维关节状态、**任务指令文本（格式待定，占位即可）**、embodiment 标识；
- 下行：32 步动作 chunk，先执行 16 步再重规划。

**安全措施**：看门狗、限位、限速、跳变检测、急停、首次 30% 限速。

**联调顺序**：假 obs → 手动牵引 → 单臂低速 → 双臂。

**通过标准**：每个联调阶段都通过。

### 8.3 评测

**目的**：量化微调后模型在真机上的表现。

每任务 20 次独立试验，记录成功/失败、时长、录屏。

**通过标准**：完成至少一轮全任务评测，结果记录归档。

**任务指令文本从哪来**：待框架确定后补充。

---

## 第 9 步：你不在实验室时能做什么

### 9.1 Windows 本机可做

- 克隆仓库、阅读代码；
- 新建 `configs/data/yam.yaml`、`configs/task/yam.yaml`（第 2、3 步）；
- **写 `tools/yam/mcap_to_lerobot.py` 并运行转换**（第 4 步）；
- **任务指令框架：保留此步骤，暂不创建具体映射文件**；
- 新建 `scripts/train_tokenizer.py` 骨架（第 6 步）；
- 下载官方权重；
- SSH 训练机。

**通过标准**：`configs/data/yam.yaml`、`configs/task/yam.yaml`、`tools/yam/mcap_to_lerobot.py`、`scripts/train_tokenizer.py` 都已在仓库里就位。

### 9.2 Windows 本机不能做

- GPU 训练；
- 真机联调。

---

## 附录 A：完整执行顺序速查

```bash
# ── Windows 本机 ──
git clone https://github.com/OpenGalaxea/GalaxeaVLA
cd GalaxeaVLA
cp configs/data/r1lite.yaml configs/data/yam.yaml      # 改 3 处
cp configs/task/r1lite.yaml configs/task/yam.yaml      # 改 3 处
mkdir -p tools/yam
# 写 tools/yam/mcap_to_lerobot.py
# 预览（不写盘）
python tools/yam/mcap_to_lerobot.py --input-root F:\ --only-task TASK-YAM-0001 --max-episodes 1 --preview
# 任务指令框架待定，暂不创建 tasks_yam.json
# 新建 scripts/train_tokenizer.py

# ── 训练机 ──
cd GalaxeaVLA
uv sync

# 5.A 下载官方权重（供第 6、7 步使用）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download OpenGalaxea/G05 --repo-type model --local-dir checkpoints

export HF_DATASETS_CACHE=/data/hf_cache
export GALAXEA_FM_OUTPUT_DIR=/data/output/g05_yam
export GALAXEA_FM_DATASET_STATS_CACHE_DIR=/data/stats_cache

# 正式转换（单条）
python tools/yam/mcap_to_lerobot.py \
  --input-root /data/yam --output-root /data/yam \
  --tasks-json tools/yam/tasks_yam.json \
  --only-task TASK-YAM-0001 --max-episodes 1 --overwrite

# 5.B 算 stats（只用 YAM 数据）
python tests/test_dataloader_batch.py \
  --mixture configs/data/yam.yaml \
  --stats data/stats/yam_stats.json \
  --stats-downsample-rate 1

# codec 微调（用官方 codec 初始化）
python scripts/train_tokenizer.py --task yam --eval-only --ckpt checkpoints/action_tokenizer.pt
python scripts/train_tokenizer.py --task yam \
    --init-ckpt checkpoints/action_tokenizer.pt \
    --out checkpoints/action_tokenizer_yam.pt \
    --freeze encoder_cb0 --epochs 2 --micro-batch 64 --grad-accum 4 --lr 5e-5
python scripts/train_tokenizer.py --task yam --eval-only --ckpt checkpoints/action_tokenizer_yam.pt

# backbone 微调（用官方 backbone 初始化）
bash scripts/run/finetune.sh 1 yam --dry-run --max_datasets 1
bash scripts/run/finetune.sh 1 yam --test --max_datasets 1 model.pretrained_ckpt=null model.max_steps=10
bash scripts/run/finetune.sh 1 yam --overfit_batch 5 model.max_steps=500
bash scripts/run/finetune.sh 8 yam logger.mode=offline

# 部署
PYTHONPATH="src:${PYTHONPATH:-}" python scripts/serve_policy.py \
  --ckpt_path $G05_OUTPUT_DIR/yam/<exp>/checkpoints/step_xxxxx/model_state_dict.pt \
  --host 0.0.0.0 --port 8080 \
  eval_embodiment=galaxea_r1lite \
  model.model_weights_to_bf16=true \
  model.model_arch.attn_implementation=sdpa \
  model.model_arch.discrete_action=true \
  model.model_arch.continuous_action=false
```

---

## 附录 B：常见坑

1. **YAM 与 R1Lite 形状同族**：不新建 parts_meta、不注册新 embodiment。
2. **action 口径必须是控制器实际下发的绝对关节目标**（弧度制）。
3. **stats、codec、backbone 必须在同一个归一化空间**。
4. **codec 码本是 EMA buffer**，冻结要连 EMA 更新一起禁。
5. **codec 微调后 token 语义会变**，backbone **必须全参微调**。
6. **数据只放 `data/yam/`**，格式由你转换时保证。
7. **训练输出 run 目录里的 sidecar 不要挪**。
8. **官方 ActionCodec 训练脚本未开源**，`scripts/train_tokenizer.py` 需自己写。
9. **官方 GalaxeaLeRobotToolkit 不能用于 YAM**，`tools/yam/mcap_to_lerobot.py` 需自写。
10. **任务指令文本必须训练和部署一致**，具体框架待定，但字段不能缺。
11. **不做 CoT**：论文主要评估也用 no-CoT 格式。
12. **`--input-root` 传 `TASK-YAM-XXXX` 的上一级目录**，不是任务目录本身。
13. **相机 CHW 布局**：脚本处理正确，不用改。
14. **`robot_type` 不影响训练**：代码里只做存取，没有白名单校验。
15. **官方权重只用于第 6、7 步微调初始化**，跟第 5.B 步算 stats 无关。

---

## 附录 C：需要找人确认的信息

1. **任务指令框架**：语言、格式、是否用映射文件、转换时如何填充 `task` 字段——**待定**；
2. YAM 控制器 SDK：关节顺序、单位（确认弧度）、下发接口、状态上报频率、夹爪极性——第 8 步客户端要用；
3. 训练机规格与 `$G05_OUTPUT_DIR` 路径；
4. 数据方确认原始 mcap 的汇聚路径和格式；
5. **部署时任务指令文本由谁指定**：操作员、上层调度，还是固定字符串——待框架确定后明确。

---

## 你现在就能做的三件事

1. 在 Windows 本机克隆仓库，复制 `r1lite.yaml` → `yam.yaml` 两份配置，按第 2、3 步改 3 处。
2. **写 `tools/yam/mcap_to_lerobot.py`，并在已有的 mcap 上跑单任务、少量 episode 的预览和转换**（第 4 步）。
3. 通读 `src/g05/tokenizer/models/actioncodec2_v2/wrapper.py`，把 wrapper 的所有公开属性/方法列出来，为写 `scripts/train_tokenizer.py` 做准备。