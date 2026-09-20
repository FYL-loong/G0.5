#!/usr/bin/env python3
"""
ActionCodec 微调脚本 —— G0.5 X-Trainer 复现 · 步骤二（自写训练循环）

自监督继续训练官方 ActionCodec（RVQ-VAE），让码本适配 X-Trainer 的动作分布。
loss = reconstruction_loss_weight * recon_MSE + commitment_loss_weight * commit_loss

⚠️ 放在 GalaxeaVLA 仓库根目录（或 scripts/ 下）运行，PYTHONPATH 需指向仓库根，
   因为要 `from g05.tokenizer...` 导入。

前置：
    - 只需要 DOBOT 采集的 observation/{n}.pkl 目录（load_actions 现场处理，无需先转 LeRobot）。
    - 数据格式已确认（见 PLAN.md）：control(14) = [left_arm(7) | right_arm(7)]、
      gripper_position(2)、control[t] ≈ joint_positions[t+1]（关节空间 replay）。
    - 归一化 stats 默认现场算；要和 backbone 训练严格一致时，先算好 dataset_stats 再用 --stats 传入。

关键事实（务必理解）：
    - gripper 是 rule-based 的（rule_based_key_patterns: [gripper]），不走 VQ-VAE，
      被 `_training_forward` 自动过滤。真正训的是 left_control(9)+right_control(9)=18 维。
    - 输入必须是【归一化后】的动作；codec 内部不再做 z-score（wrapper.py 已移除）。
    - 用官方 checkpoint 初始化 + 小 LR 继续训，别让 codebook 从零重学（防坍缩）。
    - ⚠️ 测试数据里 gripper 恒为 1.0（采集时没动过）；真训练数据需让夹爪开合，否则这 2 维学了等于没学。

用法：
    # 直接从 DOBOT 采集的 observation 目录读 pkl（推荐）
    python train_actioncodec.py \
        --data /path/to/observation \
        --out checkpoints/action_tokenizer_xtrainer.pt \
        --lr 1e-4 --epochs 10 --batch-size 64

    # 或传预处理好的 (N,32,20) .pt/.npy（跳过 pkl 处理）
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

from g05.tokenizer.models.actioncodec2_v2.wrapper import ActionCodecV2Wrapper


# ---------------------------------------------------------------------------
# 1. 构建 wrapper（复用官方 actioncodec.yaml 的 vq_config，加载官方权重）
# ---------------------------------------------------------------------------
def build_wrapper(ckpt_dir: str, device: str = "cuda") -> ActionCodecV2Wrapper:
    # actioncodec.yaml 顶层是 VQActionTokenizer，这里只取它的 vq_config 来直接构建底层 VQ-VAE
    cfg = OmegaConf.load("configs/tokenizer/actioncodec.yaml")
    vq_config = OmegaConf.to_container(cfg.vq_config, resolve=True)

    # 指向官方权重；eval=True 会让 _init_from_cfg 自动 load_model(ckpt_dir)
    vq_config["ckpt_dir"] = ckpt_dir
    vq_config["eval"] = True
    vq_config["device"] = device

    wrapper = ActionCodecV2Wrapper(vq_config)
    wrapper.model.train()  # 官方以 eval 载入，这里切回训练模式
    print(f"[codec] key_dims = {wrapper.key_dims}")
    print(f"[codec] params  = {sum(p.numel() for p in wrapper.model.parameters()) / 1e6:.2f} M")
    return wrapper


# ---------------------------------------------------------------------------
# 2. 数据准备：直接从 DOBOT pkl 读 → 切窗 → 归一化 → merge 成 (N,32,20)
# ---------------------------------------------------------------------------
# X-Trainer 20 维布局（已从采集数据确认，见 PLAN.md）：
#   control(14) = [left_arm(7) | right_arm(7)]          ← 关节空间，control[t]≈joint_positions[t+1]
#   gripper_position(2) = [left_gripper | right_gripper]
#   合并后: left_control(9) | left_gripper(1) | right_control(9) | right_gripper(1)
ARM_DOF = 7          # 每臂关节数（已确认 7 轴，不是 6/8）
CONTROL_PAD = 2      # left_arm 7 → left_control 9 补 2 个零


def _read_raw_actions_from_pkl(obs_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """读 obs_dir 下所有 {n}.pkl，返回 (control (T,14), gripper (T,2))。"""
    import glob
    import os
    import pickle

    files = sorted(
        glob.glob(os.path.join(obs_dir, "*.pkl")),
        key=lambda f: int(os.path.basename(f)[:-4]),
    )
    assert files, f"在 {obs_dir} 下没找到 pkl 文件"
    controls, grippers = [], []
    for f in files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        controls.append(np.asarray(d["control"], dtype=np.float32))
        grippers.append(np.asarray(d["gripper_position"], dtype=np.float32))
    control = torch.from_numpy(np.stack(controls))   # (T, 14)
    gripper = torch.from_numpy(np.stack(grippers))   # (T, 2)
    print(f"[data] {len(files)} 帧, control={tuple(control.shape)}, gripper={tuple(gripper.shape)}")
    return control, gripper


def _chunk(x: torch.Tensor, horizon: int, stride: int) -> torch.Tensor:
    """(T, D) → (N, horizon, D)，尾部不足一个完整窗口的帧丢弃。"""
    n = (x.shape[0] - horizon) // stride + 1
    assert n > 0, f"帧数 {x.shape[0]} 不足以切出 horizon={horizon} 的窗口"
    idx = torch.arange(horizon) + torch.arange(n).unsqueeze(1) * stride
    return x[idx]  # (N, horizon, D)


def load_actions(
    data_path: str,
    device: str,
    horizon: int = 32,
    stride: int = 8,
    stats_path: str | None = None,
) -> torch.Tensor:
    """
    返回形状 (N, T=32, D=20) 的【归一化 + merge 后】动作张量。

    两种输入：
      - data_path 指向 .pt/.npy → 直接加载（已是预处理好的 (N,32,20)）
      - data_path 指向 observation/ 目录 → 从 pkl 现场处理：
            control(14) + gripper(2)
          → 切窗 (N,32,16)
          → 归一化（z-score，只归一 control 的 14 维；gripper 保持原始值）
          → merge: left_arm(7)→left_control(9) pad2 | gripper | right_arm(7)→right_control(9) pad2 | gripper
          → (N,32,20)

    归一化 stats 默认从数据现场算；给 stats_path 则读 JSON 复用（保证与 backbone 训练一致）。
    """
    if data_path.endswith(".pt"):
        data = torch.load(data_path, map_location=device, weights_only=False)
    elif data_path.endswith(".npy"):
        data = torch.from_numpy(np.load(data_path)).to(device)
    else:
        # 目录：从 pkl 现场处理
        control, gripper = _read_raw_actions_from_pkl(data_path)   # (T,14), (T,2)

        # z-score stats（只在 control 的 14 维上算；gripper 是 rule-based 二值化，不归一）
        if stats_path is not None:
            with open(stats_path) as fp:
                stats = json.load(fp)
            mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
            std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
        else:
            mean = control.mean(dim=0)
            std = control.std(dim=0).clamp(min=1e-6)
        print(f"[data] z-score mean={mean.tolist()}")
        print(f"[data]        std ={std.tolist()}")

        control = (control - mean) / std

        # 切窗
        control = _chunk(control, horizon, stride)   # (N,32,14)
        gripper = _chunk(gripper, horizon, stride)   # (N,32,2)

        # merge 到 20 维
        left_ctrl = F.pad(control[..., 0:ARM_DOF], (0, CONTROL_PAD))                 # (N,32,9)
        right_ctrl = F.pad(control[..., ARM_DOF:2 * ARM_DOF], (0, CONTROL_PAD))     # (N,32,9)
        left_grip = gripper[..., 0:1]
        right_grip = gripper[..., 1:2]
        data = torch.cat([left_ctrl, left_grip, right_ctrl, right_grip], dim=-1)    # (N,32,20)
        print(f"[data] chunk N={data.shape[0]}, shape={tuple(data.shape)}")

    data = data.to(device)
    assert data.ndim == 3, f"期望 (N, T, D) 三维张量，得到 {data.shape}"
    assert data.shape[-1] == 20, (
        f"期望 D=20（left_control 9+left_gripper 1+right_control 9+right_gripper 1），得到 {data.shape[-1]}"
    )
    return data


# ---------------------------------------------------------------------------
# 3. 训练循环
# ---------------------------------------------------------------------------
def train(
    wrapper: ActionCodecV2Wrapper,
    actions: torch.Tensor,
    out_path: str,
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
    max_steps_per_epoch: int | None = None,
):
    n = actions.shape[0]
    dataset = TensorDataset(actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 只优化 codec 自身的可训练参数（RVQ codebook 走 EMA，不在这里；但 commitment 需要梯度）
    optimizer = torch.optim.AdamW(wrapper.model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    global_step = 0
    for epoch in range(epochs):
        for batch_idx, (act_batch,) in enumerate(loader):
            if max_steps_per_epoch is not None and batch_idx >= max_steps_per_epoch:
                break

            act_batch = act_batch.to(device)
            # _step/_max_steps 只在 consistency_loss_weight>0 时才有意义；默认 0.0，这里按约定传
            batch = {
                "action": act_batch,          # (B, T, 20)
                "_step": global_step,
                "_max_steps": max(1, epochs * len(loader)),
            }

            # forward 返回 (loss, log_dict)；log_dict 含 recon/commit/codebook 利用率
            loss, log_dict = wrapper.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            # 可选：torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), 1.0)
            optimizer.step()

            global_step += 1

            if global_step % 50 == 0:
                util = " ".join(
                    f"L{k}={log_dict.get(f'codebook/utilization_l{k}', float('nan')):.2f}"
                    for k in range(4)
                )
                print(
                    f"[step {global_step:>6}] loss={log_dict.get('loss', float(loss)):.5f} "
                    f"recon={log_dict.get('reconstruction_loss', float('nan')):.5f} "
                    f"commit={log_dict.get('commitment_loss', float('nan')):.5f} "
                    f"| codebook 利用率 {util}"
                )

        scheduler.step()

        # 每 epoch 存一次 checkpoint
        ckpt_path = out_path.replace(".pt", f"_ep{epoch + 1}.pt")
        wrapper.save_pretrained(ckpt_path)
        print(f"[epoch {epoch + 1}] saved -> {ckpt_path}")

    wrapper.save_pretrained(out_path)
    print(f"[done] final saved -> {out_path}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="动作数据：.pt/.npy 张量，或采集目录（含每帧 pkl）")
    parser.add_argument("--horizon", type=int, default=32, help="动作 chunk 长度（切窗）")
    parser.add_argument("--stride", type=int, default=8, help="滑窗步长")
    parser.add_argument("--stats", default=None, help="可选：归一化 stats JSON（与 backbone 训练一致用）")
    parser.add_argument("--ckpt", default="checkpoints/action_tokenizer.pt", help="官方 ActionCodec 权重")
    parser.add_argument("--out", default="checkpoints/action_tokenizer_xtrainer.pt")
    parser.add_argument("--lr", type=float, default=1e-4, help="小 LR（官方 checkpoint 初始化，别太大）")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=None, help="每 epoch 步数上限（调试用）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    wrapper = build_wrapper(args.ckpt, args.device)
    actions = load_actions(
        args.data, args.device,
        horizon=args.horizon, stride=args.stride, stats_path=args.stats,
    )
    train(
        wrapper, actions, args.out,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, max_steps_per_epoch=args.max_steps,
    )


if __name__ == "__main__":
    main()
