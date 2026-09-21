#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcap_to_lerobot.py
==================
把 YAM 双臂机械臂采集的 MCAP 原始数据（schema_version=2, JSON + H264）
转换成 Galaxea G0.5 训练可直接读取的 LeRobot v3.0 数据集。

数据规格（实测 episode_0001，所有话题统一 schema yam.TimestampedSample）：
  /session                         每条 1 个，含 task / 相机 / 机械臂配置
  /action/accepted   @200Hz        data.position = 14 维
                                   = left[6 关节 + 1 夹爪] ++ right[6 + 1]
                                   （训练 action 取 accepted.position，
                                     它是安全限幅后真实下发的目标）
  /left/state        @~260Hz       data.position(7) / velocity(7) / effort(7)
  /right/state       @~260Hz       同上
  /cameras/main/rgb   @30Hz        data.data = base64(H264 access unit)
  /cameras/left/rgb   @30Hz        （D405 左腕相机）
  /cameras/right/rgb  @30Hz        （D405 右腕相机）

输出特征（与官方 R1Lite 数据集命名一致，YAM 与其同为 6 轴双臂 + 双夹爪）：
  observation.images.head_rgb / left_wrist_rgb / right_wrist_rgb  (3,480,640)
  observation.state.left_arm(6) / right_arm(6) / left_gripper(1) / right_gripper(1)
  action.left_arm(6) / right_arm(6) / action.left_gripper(1) / right_gripper(1)
  task = 指令字符串（内容待定，可先用占位符）

对齐方式（与官方 GalaxeaLeRobotToolkit 一致）：
  - 以 head(main) 相机帧时戳为基准栅格（实测 30Hz）
  - 腕部相机：最近邻取帧
  - action(200Hz) / state(~260Hz)：线性插值到相机栅格，端点外推为边界值

用法（在装好 GalaxeaVLA 环境的 Linux 训练机上）：
  # 1) 不依赖 lerobot，只做解析+对齐+可视化自检（Windows 上也能跑）
  python mcap_to_lerobot.py --input-root /data/YAM --preview

  # 2) 正式转换：扫描 <input-root>/TASK-YAM-*/mcap/success/*.mcap
  #    每个任务输出一个数据集到 <output-root>/TASK-YAM-XXXX_lerobot/
  python mcap_to_lerobot.py --input-root /data/YAM --output-root data/yam \
      --tasks-json tasks_yam.json

依赖：mcap numpy pillow av(PyAV)；正式写盘还需 GalaxeaVLA 的 lerobot 环境。
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# PyAV 版本兼容：PyAV 12+ 把 av.AVError 改名为 av.error.FFmpegError，
# 旧别名被移除。统一在这里打补丁，脚本其余地方仍写 av.AVError。
# 只在真正需要时（即调用到编解码时）import av，避免 Windows 上未装 av 时报错。
# ---------------------------------------------------------------------------
def _patch_av_error():
    import av
    if not hasattr(av, "AVError"):
        av.AVError = getattr(getattr(av, "error", None), "FFmpegError", Exception)


_patch_av_error()


# ---------------------------------------------------------------------------
# 常量：话题名 / 特征名
# ---------------------------------------------------------------------------
T_ACCEPTED = "/action/accepted"
T_STATE_L = "/left/state"
T_STATE_R = "/right/state"
CAM_TOPICS = {
    "head_rgb": "/cameras/main/rgb",
    "left_wrist_rgb": "/cameras/left/rgb",
    "right_wrist_rgb": "/cameras/right/rgb",
}
CAM_KEYS = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
ARM_DOF = 6
FPS = 30


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def linear_interp(target_t, src_t, src_v):
    """与官方 interp1d(kind='linear', bounds_error=False, fill_value=边值) 等价。
    target_t: (T,)  src_t: (N,)  src_v: (N, D) 或 (N,) -> (T, D)/(T,)

    端点外推为边界值（左、右都要处理）。
    """
    target_t = np.asarray(target_t, dtype=np.float64)
    src_t = np.asarray(src_t, dtype=np.float64)
    src_v = np.asarray(src_v, dtype=np.float64)

    if len(src_t) == 0:
        raise ValueError("linear_interp: src_t is empty")
    if len(src_t) == 1:
        if src_v.ndim == 1:
            return np.full(target_t.shape, src_v[0], dtype=np.float64)
        return np.broadcast_to(src_v[0], (len(target_t),) + src_v.shape[1:]).copy()

    idx = np.searchsorted(src_t, target_t, side="left")
    idx_clip = np.clip(idx, 1, len(src_t) - 1)
    t0 = src_t[idx_clip - 1]
    t1 = src_t[idx_clip]
    v0 = src_v[idx_clip - 1]
    v1 = src_v[idx_clip]
    dt = t1 - t0
    alpha = np.zeros_like(target_t)
    nz = dt > 0
    alpha[nz] = (target_t[nz] - t0[nz]) / dt[nz]
    if src_v.ndim == 1:
        out = v0 + alpha * (v1 - v0)
    else:
        out = v0 + alpha[:, None] * (v1 - v0)

    left = target_t < src_t[0]
    if left.any():
        out[left] = src_v[0]
    right = target_t > src_t[-1]
    if right.any():
        out[right] = src_v[-1]
    return out


def nearest_index(target_t, src_t):
    """最近邻索引（官方 align_rgb 的 argmin 写法）。"""
    target_t = np.asarray(target_t)
    src_t = np.asarray(src_t)
    return np.abs(src_t[:, None] - target_t).argmin(axis=0)


# ---------------------------------------------------------------------------
# H264 解码
# ---------------------------------------------------------------------------
def _flush_codec(codec, frames):
    """把 codec 内部缓冲的帧全部取出，追加到 frames。

    不同 PyAV 版本 flush 方式不同：
      - PyAV 10+ ：用空 packet 反复 decode，直到返回空
      - 更旧版本：decode(None)
      - flush_buffers() 在部分版本返回 None，不能直接迭代
    这里把所有方式都试一遍，直到不再有新帧。
    """
    import av

    # 方式 1：空 packet 反复 decode
    got = False
    try:
        empty = av.packet.Packet(b"")
        for _ in range(1000):  # 防御性上限
            out = codec.decode(empty)
            if not out:
                break
            for f in out:
                frames.append(f.to_ndarray(format="rgb24"))
            got = True
        if got:
            return
    except (TypeError, av.AVError, ValueError):
        pass

    # 方式 2：decode(None)（旧版 PyAV）
    try:
        out = codec.decode(None)
        if out:
            for f in out:
                frames.append(f.to_ndarray(format="rgb24"))
            return
    except (TypeError, av.AVError, ValueError):
        pass

    # 方式 3：flush_buffers()，如果它返回可迭代对象就用
    try:
        result = codec.flush_buffers()
        if result is not None:
            for f in result:
                frames.append(f.to_ndarray(format="rgb24"))
    except (AttributeError, TypeError, av.AVError):
        pass


def decode_h264_sequence(au_list):
    """把一组按时间顺序的 H264 access unit（关键帧带 SPS/PPS）解码为 RGB ndarray。
    采集配置 b_frames=0，无重排序，第 i 个输出帧对应第 i 条相机消息。"""
    import av

    codec = av.codec.CodecContext.create("h264", "r")
    frames = []
    for au in au_list:
        try:
            for frame in codec.decode(av.packet.Packet(au)):
                frames.append(frame.to_ndarray(format="rgb24"))
        except av.AVError:
            # 个别 AU 解码失败时跳过，不中断整个序列
            pass
    _flush_codec(codec, frames)
    return frames


# ---------------------------------------------------------------------------
# 单个 mcap 解析
# ---------------------------------------------------------------------------
def parse_mcap(mcap_path: Path):
    """读取一个 mcap，返回对齐到 head 相机 30Hz 栅格的 dict。失败抛异常。"""
    from mcap.reader import make_reader

    bags = {t: [] for t in [T_ACCEPTED, T_STATE_L, T_STATE_R]}
    cam_msgs = {k: [] for k in CAM_TOPICS}  # (src_ns, au_bytes)
    session = None

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            topic = channel.topic
            obj = json.loads(message.data.decode("utf-8"))
            src_ns = obj["source_time_ns"]
            d = obj.get("data", obj)
            if topic == "/session":
                session = d
            elif topic in bags:
                bags[topic].append((src_ns, np.asarray(d["position"], dtype=np.float64)))
            elif topic in CAM_TOPICS.values():
                cam_key = next(k for k, v in CAM_TOPICS.items() if v == topic)
                if d.get("encoding", "h264") != "h264":
                    raise RuntimeError(f"{mcap_path.name} 相机编码不是 h264: {d.get('encoding')}")
                cam_msgs[cam_key].append((src_ns, base64.b64decode(d["data"])))

    if session is None:
        raise RuntimeError("缺少 /session")
    for name, msgs in cam_msgs.items():
        if len(msgs) == 0:
            raise RuntimeError(f"缺少相机话题 {name}")
    if len(bags[T_ACCEPTED]) == 0:
        raise RuntimeError("缺少 /action/accepted")

    # --- 解码三路相机 ---
    cam_frames = {}
    cam_times = {}
    for name, msgs in cam_msgs.items():
        msgs.sort(key=lambda x: x[0])
        aus = [m[1] for m in msgs]
        ts = np.array([m[0] for m in msgs], dtype=np.float64)
        frames = decode_h264_sequence(aus)
        if len(frames) != len(msgs):
            # 极少数情况下 flush 多/少一帧：从头对齐，丢弃多余尾帧
            n = min(len(frames), len(msgs))
            frames = frames[:n]
            ts = ts[:n]
        cam_frames[name] = np.stack(frames)  # (N,H,W,3) uint8
        cam_times[name] = ts

    # --- 基准栅格：head(main) 相机时戳 ---
    grid_ns = cam_times["head_rgb"]
    grid_s = (grid_ns - grid_ns[0]) / 1e9
    med_dt = np.median(np.diff(grid_s)) if len(grid_s) > 1 else 1.0 / FPS
    fps_est = int(round(1.0 / med_dt)) if med_dt > 0 else FPS
    if abs(fps_est - FPS) > 1:
        print(f"  [警告] {mcap_path.name} 相机帧率估计={fps_est}，预期 {FPS}")

    # 腕部相机最近邻对齐
    images = {}
    for name in CAM_KEYS:
        if name == "head_rgb":
            idx = np.arange(len(grid_ns))
        else:
            idx = nearest_index(grid_ns, cam_times[name])
            gap_ms = np.abs(cam_times[name][idx] - grid_ns).max() / 1e6
            if gap_ms > 0.5 * 1000 / FPS:
                print(f"  [警告] {mcap_path.name} {name} 最大对齐偏差 {gap_ms:.1f}ms")
        images[name] = cam_frames[name][idx]

    # --- action：/action/accepted.position，200Hz 线性插值到栅格 ---
    act = sorted(bags[T_ACCEPTED], key=lambda x: x[0])
    act_t = np.array([m[0] for m in act], dtype=np.float64)
    act_v = np.stack([m[1] for m in act])  # (N,14)
    if act_v.shape[1] != 14:
        raise RuntimeError(f"action 维度={act_v.shape[1]}，预期 14")
    act_grid = linear_interp(grid_ns, act_t, act_v)

    # --- state：左右臂 position(7)，~260Hz 线性插值 ---
    states = {}
    for side, topic in [("left", T_STATE_L), ("right", T_STATE_R)]:
        msgs = sorted(bags[topic], key=lambda x: x[0])
        st_t = np.array([m[0] for m in msgs], dtype=np.float64)
        st_v = np.stack([m[1] for m in msgs])
        if st_v.shape[1] != ARM_DOF + 1:
            raise RuntimeError(f"{topic} 维度={st_v.shape[1]}，预期 {ARM_DOF + 1}")
        states[side] = linear_interp(grid_ns, st_t, st_v)  # (T,7)

    T = len(grid_ns)
    out = {
        "task_id": session.get("task"),
        "fps": fps_est,
        "T": T,
        "duration_s": float(grid_s[-1]) if T > 1 else 0.0,
        # images: (T,H,W,3) uint8
        "observation.images.head_rgb": images["head_rgb"],
        "observation.images.left_wrist_rgb": images["left_wrist_rgb"],
        "observation.images.right_wrist_rgb": images["right_wrist_rgb"],
        # state (T, ...) float64
        "observation.state.left_arm": states["left"][:, :ARM_DOF],
        "observation.state.right_arm": states["right"][:, :ARM_DOF],
        "observation.state.left_gripper": states["left"][:, ARM_DOF:ARM_DOF + 1],
        "observation.state.right_gripper": states["right"][:, ARM_DOF:ARM_DOF + 1],
        # action (T, ...) float64
        "action.left_arm": act_grid[:, :ARM_DOF],
        "action.right_arm": act_grid[:, 7:7 + ARM_DOF],
        "action.left_gripper": act_grid[:, ARM_DOF:ARM_DOF + 1],
        "action.right_gripper": act_grid[:, 7 + ARM_DOF:7 + ARM_DOF + 1],
    }
    return out


# ---------------------------------------------------------------------------
# LeRobot v3.0 数据集写入
# ---------------------------------------------------------------------------
def build_features(sample_episode):
    """声明 LeRobot v3 数据集的 features。

    ⚠️ 相机 shape 这里按 r1lite 的 raw_shape [3, 480, 640]（CHW）声明。
    写入时会把 HWC 图像转成 CHW（见 write_task_dataset）。
    若训练时发现 shape 不匹配，把这里改成 (h, w, c)，并去掉转置。
    """
    h, w, c = sample_episode["observation.images.head_rgb"].shape[1:]
    img_feat = lambda: {"dtype": "video", "shape": (c, h, w),
                        "names": ["channels", "height", "width"]}
    arm_feat = lambda kind, topic: {
        "dtype": "float64", "shape": (ARM_DOF,),
        "names": [f"{topic}.{kind}[{i}]" for i in range(ARM_DOF)]}
    grip_feat = lambda name, topic: {"dtype": "float64", "shape": (1,),
                                     "names": [f"{topic}.{name}[0]"]}
    features = {
        "observation.images.head_rgb": img_feat(),
        "observation.images.left_wrist_rgb": img_feat(),
        "observation.images.right_wrist_rgb": img_feat(),
        "observation.state.left_arm": arm_feat("left", "position", "/left/state"),
        "observation.state.right_arm": arm_feat("right", "position", "/right/state"),
        "observation.state.left_gripper": grip_feat("left_gripper", "/left/state"),
        "observation.state.right_gripper": grip_feat("right_gripper", "/right/state"),
        "action.left_arm": arm_feat("left", "position", "/action/accepted"),
        "action.right_arm": arm_feat("right", "position", "/action/accepted"),
        "action.left_gripper": grip_feat("left_gripper", "/action/accepted"),
        "action.right_gripper": grip_feat("right_gripper", "/action/accepted"),
    }
    return features


def import_lerobot_writer():
    """优先用 GalaxeaVLA 自带的 vendored v3 写入器（与训练读取端严格配套）；
    退化到独立安装的 GalaxeaLerobot3.0 的 lerobot 包。返回 (类, 变体名)。"""
    try:
        from g05.data.lerobot.lerobot_dataset_v3 import LeRobotDataset as DS
        return DS, "vendored"
    except ImportError:
        pass
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as DS
    return DS, "standalone"


def write_task_dataset(task_id, episodes, instruction, out_root: Path, overwrite=False):
    """写一个任务的 LeRobot v3 数据集。

    ⚠️ robot_type 这里写 'yam_bimanual'。若训练框架对 robot_type 有白名单，
    需要改成与 r1lite 一致的值（核对 configs/data/r1lite.yaml 或官方 r1lite 数据集）。
    """
    DS, variant = import_lerobot_writer()

    repo_id = f"{task_id}_lerobot"
    root = out_root / repo_id
    if root.exists():
        if overwrite:
            import shutil
            shutil.rmtree(root)
        else:
            raise RuntimeError(f"{root} 已存在，加 --overwrite 覆盖")

    features = build_features(episodes[0])
    ds = DS.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=str(root),
        robot_type="yam_bimanual",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
    )

    for ep in episodes:
        T = ep["T"]
        for i in range(T):
            frame = {k: ep[k][i] for k in features}
            # HWC -> CHW（与 build_features 中的声明一致）
            for cam in CAM_KEYS:
                key = f"observation.images.{cam}"
                frame[key] = np.transpose(frame[key], (2, 0, 1))
            if variant == "vendored":
                frame["task"] = instruction
                ds.add_frame(frame)
            else:
                # 独立版 GalaxeaLerobot3.0：save_episode 固定取 5 个任务槽
                # [粗指令, 细粒度指令, 粗质量, 细质量, 操作手]
                ds.add_frame(frame, task=[instruction, instruction,
                                          "qualified", "qualified", "both"])
        ds.save_episode()
    print(f"  写出 {root}（{len(episodes)} episodes, {variant} writer）")


# ---------------------------------------------------------------------------
# preview 自检（不需要 lerobot）
# ---------------------------------------------------------------------------
def preview_episode(ep, out_dir: Path):
    """把 5 帧抽样拼成横向条带，存成 PNG 供肉眼检查。

    注意：parse_mcap 输出的图像是 HWC，这里直接按 HWC 处理（不转置）。
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    T = ep["T"]
    picks = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    for cam in CAM_KEYS:
        key = f"observation.images.{cam}"
        strip = np.concatenate([ep[key][p] for p in picks], axis=1)  # HWC 沿 W 拼
        p = out_dir / f"preview_{ep['task_id']}_{cam}.png"
        Image.fromarray(strip).save(p)
        print("  saved", p)
    composite = np.concatenate([ep[f"observation.images.{c}"][T // 2]
                                for c in CAM_KEYS], axis=1)
    Image.fromarray(composite).save(out_dir / f"preview_{ep['task_id']}_composite_mid.png")

    print(f"  task={ep['task_id']} T={T} fps={ep['fps']} dur={ep['duration_s']:.2f}s")
    for k in ["action.left_arm", "action.right_arm", "action.left_gripper",
              "action.right_gripper", "observation.state.left_gripper",
              "observation.state.right_gripper"]:
        v = ep[k]
        print(f"    {k:38s} shape={str(v.shape):10s} min={v.min():+.3f} max={v.max():+.3f}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def discover_mcaps(input_root: Path):
    """按 task id 分组 success 目录下的 mcap。"""
    groups = {}
    for p in sorted(input_root.rglob("*.mcap")):
        if p.parent.name != "success":
            continue
        task_id = next((part for part in p.parts if part.startswith("TASK-YAM-")), None)
        if task_id is None:
            continue
        groups.setdefault(task_id, []).append(p)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, type=Path,
                    help="包含 TASK-YAM-XXXX/mcap/success/*.mcap 的根目录")
    ap.add_argument("--output-root", type=Path, default=None,
                    help="LeRobot 数据集输出根目录（每个任务一个子目录）")
    ap.add_argument("--tasks-json", type=Path, default=None,
                    help="task id -> 指令 映射 JSON。正式转换必需；--preview 模式可省略。")
    ap.add_argument("--only-task", default=None, help="只处理某个 TASK-YAM-XXXX")
    ap.add_argument("--max-episodes", type=int, default=0, help="每个任务最多转换条数（0=全部）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview", action="store_true",
                    help="只解析+自检出图，不写 LeRobot 数据集（无需 lerobot 环境）")
    args = ap.parse_args()

    if not args.preview and args.tasks_json is None:
        sys.exit("非 preview 模式必须提供 --tasks-json")
    if not args.preview and args.output_root is None:
        sys.exit("非 preview 模式必须提供 --output-root")

    task_instr = {}
    if args.tasks_json is not None:
        if not args.tasks_json.exists():
            sys.exit(f"tasks-json 文件不存在: {args.tasks_json}")
        task_instr = json.loads(Path(args.tasks_json).read_text(encoding="utf-8"))

    groups = discover_mcaps(args.input_root)
    if not groups:
        sys.exit(f"在 {args.input_root} 下没找到 **/mcap/success/*.mcap")

    preview_dir = Path("./preview_out") if args.preview else args.output_root

    for task_id, files in sorted(groups.items()):
        if args.only_task and task_id != args.only_task:
            continue
        if not args.preview and task_id not in task_instr:
            print(f"[跳过] {task_id}：tasks_yam.json 里没有指令，请补充后再转")
            continue
        print(f"== {task_id}: 发现 {len(files)} 个 mcap ==")
        episodes = []
        for mp in files:
            if args.max_episodes and len(episodes) >= args.max_episodes:
                break
            try:
                ep = parse_mcap(mp)
                episodes.append(ep)
                print(f"   OK  {mp.name}  T={ep['T']} dur={ep['duration_s']:.1f}s")
            except Exception as e:
                print(f"   [损坏/跳过] {mp.name}: {e}")
        if not episodes:
            print("   没有可转换的完整 episode")
            continue

        if args.preview:
            preview_episode(episodes[0], preview_dir / "_preview")
            continue

        write_task_dataset(task_id, episodes, task_instr[task_id],
                           args.output_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()