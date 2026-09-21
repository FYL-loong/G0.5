#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GalaxeaVLA-main/
├── configs/
├── scripts/
│   ├── run/
│   ├── serve_policy.py
│   └── train_tokenizer.py      ← 放这里
├── src/
└── tools/
==========================
YAM 复现 G0.5 —— ActionCodec 微调脚本。

官方 ActionCodec 训练脚本未开源（内部名 train_vq.py），本文件补齐。

设计原则：
  * 不重写任何模型/损失：模型用官方 ActionCodecV2Wrapper；
  * 数据管线复用官方：instantiate_dataset / build_processors；
  * 存盘格式对齐官方 load_model：{"model_state_dict", "tokenizer_meta"}；
  * 单卡即可（codec 仅约 484MB），不做 DDP。

用法：
  # 基线评估
  python scripts/train_tokenizer.py --task yam --eval-only \
      --ckpt checkpoints/action_tokenizer.pt

  # 微调（推荐：冻结 conv_in + encoder）
  python scripts/train_tokenizer.py --task yam \
      --init-ckpt checkpoints/action_tokenizer.pt \
      --out checkpoints/action_tokenizer_yam.pt \
      --freeze encoder --epochs 2 --micro-batch 64 --grad-accum 4 --lr 5e-5

  # 评估微调后
  python scripts/train_tokenizer.py --task yam --eval-only \
      --ckpt checkpoints/action_tokenizer_yam.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

# ── 与 tests/test_dataloader_batch.py 相同的路径引导 ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# 配置加载（与 finetune.sh / test_dataloader_batch.py 一致）
# ---------------------------------------------------------------------------
def load_task_config(task_name: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from g05.utils.config.config_resolvers import register_default_resolvers

    register_default_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"),
                                version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task_name}"])
    return cfg


# ---------------------------------------------------------------------------
# 构建 dataset / processor / stats
# ---------------------------------------------------------------------------
def build_data(cfg, val_ratio: float, num_workers: int, batch_size: int):
    from g05.utils.data.processor_utils import build_processors, instantiate_dataset
    from g05.utils.data.normalizer import (
        load_dataset_stats_from_json,
        save_dataset_stats_to_json,
    )

    # r1lite 默认 val_set_proportion=1e-4 几乎切不出验证集；覆盖成 0.02
    OmegaConf.set_struct(cfg.data, False)
    cfg.data.val_set_proportion = float(val_ratio)

    train_dataset = instantiate_dataset(cfg, is_training_set=True)
    eval_dataset = instantiate_dataset(cfg, is_training_set=False)
    processor = build_processors(cfg)

    stats_path = Path(cfg.datastatics_path) if cfg.get("datastatics_path", None) else None
    if stats_path is not None and stats_path.exists() and stats_path.stat().st_size > 0:
        dataset_stats = load_dataset_stats_from_json(stats_path)
        print(f"[stats] loaded {stats_path}")
    else:
        print("[stats] 文件不存在，现算（遍历全量数据，需要一些时间）...")
        dataset_stats = train_dataset.get_dataset_stats(processor)
        if stats_path is not None:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            save_dataset_stats_to_json(dataset_stats, stats_path)
            print(f"[stats] saved -> {stats_path}")

    processor.set_normalizer_from_stats(dataset_stats)
    train_dataset.set_processor(processor)
    eval_dataset.set_processor(processor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=default_collate,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=default_collate,
    )
    print(f"[data] train={len(train_dataset)} eval={len(eval_dataset)}")
    return train_loader, eval_loader


# ---------------------------------------------------------------------------
# ActionCodec wrapper 构造
# ---------------------------------------------------------------------------
def build_codec(cfg, ckpt_path: str, device: str, freeze_mode: str):
    """构造 wrapper，加载官方权重，按冻结策略设置 requires_grad。

    冻结第 0 层 quantizer 时用 q0.eval()（不用猜 EMA 方法名）。
    训练循环里每步前会再切一次 eval，防止 model.train() 覆盖。
    """
    from g05.tokenizer.models.actioncodec2_v2.wrapper import ActionCodecV2Wrapper

    vq = OmegaConf.to_container(cfg.model.tokenizer.vq_config, resolve=True)
    vq["eval"] = True
    vq["ckpt_dir"] = ckpt_path
    vq["device"] = device
    wrapper = ActionCodecV2Wrapper(vq)

    model = wrapper.model

    def freeze_module(m):
        for p in m.parameters():
            p.requires_grad = False

    def reapply_quantizer_eval():
        """训练循环里每步前重切第 0 层 quantizer 到 eval，防止被 model.train() 覆盖。"""
        if freeze_mode == "encoder_cb0":
            try:
                model.rvq.quantizers[0].eval()
            except (AttributeError, IndexError):
                pass

    if freeze_mode == "encoder":
        # 推荐：冻结输入侧表征，训练后续所有层
        freeze_module(model.conv_in)
        freeze_module(model.encoder)
    elif freeze_mode == "encoder_cb0":
        # 更激进：冻结输入侧 + 第 0 层主码本
        freeze_module(model.conv_in)
        freeze_module(model.encoder)
        try:
            q0 = model.rvq.quantizers[0]
            freeze_module(q0)
            q0.eval()
        except (AttributeError, IndexError) as e:
            print(f"[warn] encoder_cb0：找不到 model.rvq.quantizers[0]（{e}）；"
                  f"退化为只冻结 conv_in+encoder")
    elif freeze_mode == "none":
        pass
    elif freeze_mode == "all_nn":
        freeze_module(model)
    else:
        raise ValueError(f"unknown freeze_mode: {freeze_mode}")

    wrapper.to(device)
    wrapper.model.train()
    reapply_quantizer_eval()

    n_train = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in wrapper.parameters())
    print(f"[codec] freeze={freeze_mode}  trainable={n_train/1e6:.2f}M/{n_total/1e6:.2f}M")
    print(f"[codec] key_dims={wrapper.key_dims}  nn_keys={wrapper._nn_keys}  rule_keys={wrapper._rule_keys}")
    return wrapper, reapply_quantizer_eval


# ---------------------------------------------------------------------------
# 把扁平 action 按 key_dims 拆开、丢掉规则化夹爪键
# ---------------------------------------------------------------------------
def split_nn_components(wrapper, action: torch.Tensor):
    dims = list(wrapper.key_dims.values())
    total = sum(dims)
    pieces = torch.split(action[..., :total], dims, dim=-1)
    comp = dict(zip(wrapper.key_dims.keys(), pieces))
    return {k: v for k, v in comp.items() if not wrapper.is_rule_based_key(k)}


# ---------------------------------------------------------------------------
# 评估：归一化空间逐键 RMSE + 码本利用率
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(wrapper, loader, device, max_batches: int, reapply_eval=None):
    wrapper.model.eval()
    key_sse, key_n = {}, {}
    util_sum, util_cnt = {}, {}
    n_batches = 0

    for batch in loader:
        x = batch["action"].float().to(device)
        comp = split_nn_components(wrapper, x)

        # 直接调 model（component_dict 无 "action" 键，走 eval 分支）
        # 返回 (recon_dict, codes_dict, loss_dict)
        recon_dict, codes_dict, loss_dict = wrapper.model(
            comp, d_original=wrapper.key_dims
        )

        for k, v in comp.items():
            d_k = wrapper.key_dims[k]
            err = recon_dict[k][..., :d_k] - v[..., :d_k]
            key_sse[k] = key_sse.get(k, 0.0) + float((err ** 2).sum())
            key_n[k] = key_n.get(k, 0) + err.numel()

        for k, v in loss_dict.items():
            if k.startswith("codebook/utilization_l"):
                lvl = int(k.rsplit("_l", 1)[1])
                util_sum[lvl] = util_sum.get(lvl, 0.0) + float(v)
                util_cnt[lvl] = util_cnt.get(lvl, 0) + 1

        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    metrics = {f"rmse_{k}": (key_sse[k] / max(key_n[k], 1)) ** 0.5 for k in key_sse}
    for lvl in sorted(util_sum):
        metrics[f"utilization_l{lvl}"] = util_sum[lvl] / max(util_cnt[lvl], 1)

    wrapper.model.train()
    if reapply_eval is not None:
        reapply_eval()
    return metrics


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "codec 微调请在 GPU 机器上运行"
    torch.manual_seed(args.seed)

    cfg = load_task_config(args.task)
    train_loader, eval_loader = build_data(
        cfg, val_ratio=args.val_ratio, num_workers=args.workers,
        batch_size=args.micro_batch,
    )
    wrapper, reapply_eval = build_codec(cfg, args.init_ckpt, device, args.freeze)

    params = [p for p in wrapper.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    try:
        from transformers import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
        have_sched = True
    except Exception:
        have_sched = False

    log_f = open(args.log_file, "a", encoding="utf-8") if args.log_file else None
    global_step = 0
    print(f"[train] steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup_steps}")

    for epoch in range(args.epochs):
        opt.zero_grad()
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            # 每步前重切冻结的 quantizer 到 eval，防止被 wrapper 内 model.train() 覆盖
            reapply_eval()

            x = batch["action"].float().to(device)
            loss, log_dict = wrapper(
                {"action": x, "_step": global_step, "_max_steps": total_steps}
            )
            (loss / args.grad_accum).backward()

            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if have_sched:
                    sched.step()
                opt.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    msg = {
                        "epoch": epoch, "step": global_step,
                        "lr": opt.param_groups[0]["lr"],
                        **{k: v for k, v in log_dict.items()
                           if k in ("loss", "reconstruction_loss", "commitment_loss")
                           or k.startswith("recon/") or k.startswith("codebook/")},
                    }
                    line = json.dumps(msg, ensure_ascii=False)
                    print(f"ep{epoch} step{global_step}/{total_steps} {line}")
                    if log_f:
                        log_f.write(line + "\n")
                        log_f.flush()

            if args.max_steps and global_step >= args.max_steps:
                break

        print(f"[train] epoch {epoch} done in {time.time()-t0:.0f}s")
        m = evaluate(wrapper, eval_loader, device, args.eval_batches, reapply_eval)
        print("[eval] " + json.dumps(m, ensure_ascii=False))
        if log_f:
            log_f.write(json.dumps({"epoch_eval": epoch, **m},
                                   ensure_ascii=False) + "\n")
            log_f.flush()

    # 存盘（与官方 load_model 对齐：{"model_state_dict", "tokenizer_meta"}）
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": wrapper.state_dict(),
            "tokenizer_meta": {"parts_meta": dict(wrapper.key_dims)},
        },
        args.out,
    )
    print(f"[save] {args.out}")
    if log_f:
        log_f.close()


def eval_only(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_task_config(args.task)
    _train_loader, eval_loader = build_data(
        cfg, val_ratio=args.val_ratio, num_workers=args.workers,
        batch_size=args.micro_batch,
    )
    wrapper, _ = build_codec(cfg, args.ckpt, device, freeze_mode="all_nn")
    m = evaluate(wrapper, eval_loader, device, args.eval_batches)
    print(f"[eval-only] ckpt={args.ckpt}")
    print(json.dumps(m, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ActionCodec on YAM (G0.5)")
    p.add_argument("--task", default="yam", help="configs/task/<task>.yaml")
    p.add_argument("--init-ckpt", default="checkpoints/action_tokenizer.pt")
    p.add_argument("--out", default="checkpoints/action_tokenizer_yam.pt")
    p.add_argument("--ckpt", default="checkpoints/action_tokenizer.pt",
                   help="--eval-only 时评估的权重")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--freeze", dest="freeze_mode", default="encoder",
                   choices=["none", "encoder", "encoder_cb0", "all_nn"],
                   help="默认 encoder：冻结 conv_in+encoder，最稳；"
                        "encoder_cb0 额外冻结第 0 层主码本")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--micro-batch", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=50, help="0=全量验证集")
    p.add_argument("--max-steps", type=int, default=0, help="调试用，>0 限制步数")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-file", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        eval_only(args)
    else:
        train(args)