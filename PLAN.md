# G0.5 在 DOBOT X-Trainer 上的复现计划（微调路线）

> 目标：在 **DOBOT X-Trainer** 上复现 G0.5 的动作控制能力。
> 路线：**微调官方 ActionCodec + 微调官方 G0.5 backbone**（不是从零训，也不用自己预训练）。
> 依据：论文 arXiv:2608.11739v1 + 仓库 https://github.com/OpenGalaxea/GalaxeaVLA

---

## 0. 总体路线（先看清全貌）

```
① 采集 X-Trainer 遥操数据 → 转 LeRobot v3.0 格式
② 算 X-Trainer 自己的 normalizer stats（dataset_stats.json）
③ 微调 ActionCodec（用官方 checkpoint 初始化，自监督继续训，冻结前先训好）
④ 微调官方 G0.5 backbone（用微调后的 codec，全参微调）
⑤ 部署推理（serve_policy.py，chunk + receding horizon）
```

**两个关键决策（已确认）：**

| 组件 | 决策 | 理由 |
|---|---|---|
| **ActionCodec** | ✅ 微调权重 | 牺牲跨形态通用性（你只有 1 台机器人用不上），换取对 X-Trainer 动作的更高重建精度 |
| **backbone** | ✅ 全参微调 | 官方 backbone 已预训练，微调让它适配 X-Trainer + 新 codec 的 token 语义 |

---

## 1. 架构速览：X-Trainer 如何映射进 G0.5

G0.5 = Qwen3.5 2B（原生多模态 VLM）+ ActionCodec（动作 tokenizer）+ 视觉记忆。

- **Qwen3.5 自带**文本 tokenizer + ViT 视觉塔（图像→连续 patch embedding），无需额外装。
- **ActionCodec**（RVQ-VAE，4 层码本 × 4096 码 × 8 维）把连续动作编成 32 个离散 token（每 chunk）。训练用 encoder（产标签），推理用 decoder（解回动作）。
- **动作空间**：统一 27 维 `<left_control>(9)|<left_gripper>(1)|<right_control>(9)|<right_gripper>(1)|<lower_body>(7)`。

**X-Trainer 的映射**：X-Trainer = 双臂 7 轴 Nova2 ×2 + 2 夹爪 + 3 相机（头 + 双腕），**固定底座，无下半身**。（已从采集数据确认：`joint_positions` = 14 = 2×7，双臂各 7 轴。）

- 7 轴 ≤ 9，直接 pad 到 `left_control(9)`；无 lower_body → 用 **20 维**布局（去掉 lower_body 的 7）。
- 对应仓库里的 `configs/data/parts_meta/dual_arm_grouped_0409.yaml`：

```
left_control(9) | left_gripper(1) | right_control(9) | right_gripper(1)  = 20 维
```

> 结论：X-Trainer 干净映射进 20 维空间，无需改 codec 的 `max_component_dim`。但注意：官方 `dual_arm_grouped_0409.yaml` 的 `left_arm: 8` 是别的机器人；X-Trainer 是 7 轴，要**自建 parts_meta**（`left_arm: 7, right_arm: 7`，`left_control` 仍 = 9 靠 pad）。

---

## 2. 前置：数据采集与 LeRobot 格式

1. **遥操采集**：DOBOT X-Trainer 双臂遥操，记录 3 路 RGB（头 + 双腕）+ 双臂关节角 + 双夹爪状态，25Hz。
2. **转 LeRobot v3.0 格式**（`lerobot_ds_version: '3.0'`）：
   - `observation.state.left_arm` / `right_arm`（各 7 维）
   - `observation.state.left_gripper` / `right_gripper`（各 1 维）
   - `action.left_arm` / `right_arm` / `left_gripper` / `right_gripper`
   - `observation.images.head_rgb` / `left_wrist_rgb` / `right_wrist_rgb`
3. **新建 `configs/data/xtrainer.yaml`**，模板用 `configs/data/r1pro.yaml`（双臂 + 夹爪结构最接近，X-Trainer 和 r1pro 都是 7 轴），改两处：
   - ~~双臂 `raw_shape` 从 `7` 改成 `6`~~ → **raw_shape 已是 7，无需改**
   - 删除 `lower_body`/`torso` 相关项
   - 相机名改成 X-Trainer 的 `head_rgb` / `left_wrist_rgb` / `right_wrist_rgb`

---

## 3. 步骤一：算 normalizer stats

ActionCodec 工作在归一化空间，normalizer 是**每个数据源单独算**的（z-score-tail / q01-q99），不是训出来的。

```bash
python tests/test_dataloader_batch.py \
  --mixture configs/data/xtrainer.yaml \
  --stats ./dataset_stats_xtrainer.json \
  --stats-downsample-rate 1
```

产物 `dataset_stats_xtrainer.json` 在训练时通过 `datastatics_path` 引用。

---

## 4. 步骤二：微调 ActionCodec

> ⚠️ **重要**：本仓库（GalaxeaVLA）**没有 ActionCodec 训练脚本**。codec 的模型 + loss 代码在 `src/g05/tokenizer/models/actioncodec2_v2/`（`wrapper.py`、`modeling_actioncodec2v2.py`），但训练循环在另一个仓库（g05-base 预训练流水线）。需二选一：
> - **A（推荐）**：自写一个 codec 微调循环，用官方 `checkpoints/action_tokenizer.pt` 初始化，在 X-Trainer 动作数据上继续训。
> - **B**：找到官方发布 ActionCodec 训练脚本的独立仓库，直接复用。

### 4.1 自写训练循环的要点

1. **加载**：`ActionCodecV2Wrapper` 从 `checkpoints/action_tokenizer.pt` 加载权重（`configs/tokenizer/actioncodec.yaml` 是它的配置，`parts_meta` 保持 20 维）。
2. **数据**：归一化后的 X-Trainer 动作序列，形状 `(B, 32, 20)`（32 步 chunk × 20 维）。
3. **损失**（自监督，无标签）：`modeling_actioncodec2v2.py` 里 forward 已返回：
   ```
   loss = reconstruction_loss_weight * recon_MSE + commitment_loss_weight * commit_loss
   ```
4. **关键**：用官方 checkpoint 初始化 + **小 LR**（比从头训低 1~2 个数量级，如 1e-4 量级起步）继续训，别让 codebook 从零重学（否则坍缩 + 丢失全部预训练语义）。
5. **保存**：产出新的 `action_tokenizer_xtrainer.pt`。

### 4.2 微调 codec 的代价（已确认接受）

微调后码本变了 → **token 语义变了** → 官方 backbone 的「动作头」错位。这没关系，因为第 5 步会**全参微调 backbone**，动作头会重新学。被打乱的只是「动作解码」这部分；**视觉/语言先验保留**（用户判断正确）。

---

## 5. 步骤三：微调官方 backbone

### 5.1 新建 `configs/task/xtrainer.yaml`

模板用 `configs/task/so100.yaml`（6 轴单臂）或 `r1pro.yaml`（双臂），核心改动：

```yaml
defaults:
  - override /model: g05
  - override /tokenizer: actioncodec
  - override /data: xtrainer
  - _self_

datastatics_path: ./dataset_stats_xtrainer.json   # 第 3 步产物

model:
  model_arch:
    action_dim: 20          # 20 维（无 lower_body）
    proprio_dim: 20
  processor:
    action_state_merger:
      _target_: g05.data_processor.transforms.action_state_merger.GroupedPaddingMerger
      max_action_shape_meta: ${oc.load:configs/data/parts_meta/dual_arm_grouped_0409.yaml,parts_meta}
      max_state_shape_meta:  ${oc.load:configs/data/parts_meta/dual_arm_grouped_0409.yaml,parts_meta}
      merge_spec:            ${oc.load:configs/data/parts_meta/dual_arm_grouped_0409.yaml,merge_spec}
      merge: true

tokenizer:
  vq_config:
    ckpt_dir: ./action_tokenizer_xtrainer.pt    # ← 指向第 4 步微调后的 codec
    parts_meta: {left_control: 9, left_gripper: 1, right_control: 9, right_gripper: 1}
```

### 5.2 训练超参（参考值，可调）

| 项 | 值 | 说明 |
|---|---|---|
| `learning_rate` | 4e-5 ~ 8e-5 | 参考 so100 的 8e-5 / r1pro 的 4e-5 |
| `batch_size` | 8 ~ 16 | 按显存定 |
| `max_epochs` | 2 ~ 10 | 单机器人数据量小，几 epoch 够 |
| `warmup_steps` | 200 ~ 1000 | |
| `vla_training_strategy` | `vla-full-train` | 全参微调（动作头必须重学，别用冻结） |
| `vision_lr_multiplier` | 1.0（或 0.1） | 视觉塔可适当降 LR |

### 5.3 启动训练

```bash
# 单机 N 卡
bash scripts/run/finetune.sh <num_gpus> xtrainer

# 快速冒烟测试（离线日志、截断数据、1 步）
bash scripts/run/finetune.sh 1 xtrainer --test

# 只看配置是否正确解析（不训练）
bash scripts/run/finetune.sh 1 xtrainer --dry-run
```

训练时，processor 会用 `ckpt_dir` 指向的（微调后）codec 在线把真值动作编成 token，作为 next-token CE 的目标。backbone 在微调中重新学动作头，适配新 codec 的 token 语义。

---

## 6. 步骤四：部署推理

`scripts/serve_policy.py` 提供 chunk 循环 + receding horizon：

- `action_steps`（默认 16，`1` = 每步重算 RTC）：预测 32 步 chunk，执行 K 步，再拿新观测重规划。
- 推理只用 codec 的 **decoder**（预测 token → 连续动作），encoder 不再用。
- 训练侧 copy 出来的 `action_tokenizer.pt` 会随输出目录一起保存，serve 时直接加载。

---

## 7. 风险与注意事项

1. **codec 微调后 token 语义变化** → backbone 必须**全参微调**（不能只 LoRA 动作头就完事，但全参是默认，已覆盖）。
2. **codebook 坍缩风险**：微调 codec 时 RVQ 的码可能死掉（dead code），注意监控 codebook 利用率；用小 LR + EMA 缓解。
3. **数据量**：单机器人微调不需要 14 个 embodiment 的海量数据，但**任务要多样**（语言跟随需要多样指令）。建议先 50h 量级，参考论文 R1-Lite 50h。
4. **no CoT**：默认 `predict_cot: false`，动作控制这条线不依赖 CoT，先别管（见「与论文路线区别」）。
5. **`parts_meta` 三处必须一致**：codec 的 `vq_config.parts_meta` ↔ processor 的 `action_state_merger` ↔ data 配置的 shape_meta，否则维度对不上直接报错。

---

## 8. 与「从零复现论文」路线的区别（备忘）

| | 从零复现论文 | **本计划（X-Trainer 微调）** |
|---|---|---|
| ActionCodec | 从零训（需多 embodiment 动作数据） | **微调官方 checkpoint** |
| Qwen backbone | 从 Qwen3.5 基座预训练成 VLA（14 embodiment + 多卡） | **微调官方预训练好的 backbone** |
| CoT | 要搭 autolabeling 流水线 | **跳过**（`predict_cot: false`） |
| 数据 | 14 个 embodiment 海量遥操 + VQA | **单台 X-Trainer 遥操数据** |
| 硬件 | 多卡集群 | 单机可做 |

一句话：**本计划 = 微调 codec + 微调 backbone，两条线都用官方 checkpoint 初始化，不做从零预训练。**
