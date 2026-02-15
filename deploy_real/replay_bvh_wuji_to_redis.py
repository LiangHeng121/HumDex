#!/usr/bin/env python3
"""
Replay BVH motion to Wuji hands via Redis (WUJI ONLY).

做什么：
- 读取 BVH（与 replay_bvh_body_to_redis.py 相同的 loader + 命名映射）
- 从 BVH 的手部骨骼（LeftHand/LeftIndexFinger...）构造 Wuji server 需要的 26D hand_tracking 字典：
    key 示例: "LeftHandWrist": [[x,y,z], [qw,qx,qy,qz]]
  注意：Wuji server 实际只用 [x,y,z] 位置，quat 可用单位四元数占位。
- 写入 Redis：
    - hand_tracking_left/right_unitree_g1_with_hands
    - wuji_hand_mode_left/right_unitree_g1_with_hands = follow

不做什么：
- 不写 action_body/action_neck/action_hand_*（不控制全身/脖子/Unitree 夹爪手）

使用前提：
- 你已经在 g1 上启动 Wuji 控制器（server_wuji_hand_redis.py 或 server_wuji_hands_redis_dual.py），并连到同一个 Redis。
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis


def _now_ms() -> int:
    return int(time.time() * 1000)


class _RateLimiter:
    def __init__(self, hz: float):
        self.dt = 1.0 / max(1e-6, float(hz))
        self.next_t = time.monotonic()

    def sleep(self):
        self.next_t += self.dt
        now = time.monotonic()
        delay = self.next_t - now
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_t = now


def _transform_points(
    pts: Dict[str, np.ndarray],
    origin: Optional[np.ndarray],
    scale: float,
) -> Dict[str, np.ndarray]:
    """
    Translate by origin (if provided) and scale points (for visualization only).
    """
    out: Dict[str, np.ndarray] = {}
    for k, p in pts.items():
        q = np.asarray(p, dtype=np.float32).reshape(3)
        if origin is not None:
            q = q - origin
        q = q * float(scale)
        out[k] = q
    return out


def _apply_fixed_range(ax, r: float):
    ax.set_xlim3d(-r, r)
    ax.set_ylim3d(-r, r)
    ax.set_zlim3d(-r, r)


def _collect_bvh_hand_joint_points(frame: Dict[str, Any], side: str) -> Dict[str, np.ndarray]:
    """
    收集 BVH 手部 JOINT（不含 End Site），用于可视化。
    """
    side = side.lower()
    assert side in ["left", "right"]
    pfx = "Left" if side == "left" else "Right"
    names = [
        f"{pfx}Hand",
        f"{pfx}ThumbFinger",
        f"{pfx}ThumbFinger1",
        f"{pfx}ThumbFinger2",
        f"{pfx}IndexFinger",
        f"{pfx}IndexFinger1",
        f"{pfx}IndexFinger2",
        f"{pfx}IndexFinger3",
        f"{pfx}MiddleFinger",
        f"{pfx}MiddleFinger1",
        f"{pfx}MiddleFinger2",
        f"{pfx}MiddleFinger3",
        f"{pfx}RingFinger",
        f"{pfx}RingFinger1",
        f"{pfx}RingFinger2",
        f"{pfx}RingFinger3",
        f"{pfx}PinkyFinger",
        f"{pfx}PinkyFinger1",
        f"{pfx}PinkyFinger2",
        f"{pfx}PinkyFinger3",
    ]
    pts: Dict[str, np.ndarray] = {}
    for n in names:
        if n in frame:
            pts[n] = _get_pos(frame, n).reshape(3)
    return pts


def _collect_bvh_hand_end_sites(frame: Dict[str, Any], side: str, endsite_offsets_cm: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    计算该 hand side 的 End Site 点（用于可视化）。
    """
    side = side.lower()
    assert side in ["left", "right"]
    pfx = "Left" if side == "left" else "Right"
    end_pts: Dict[str, np.ndarray] = {}
    for joint in endsite_offsets_cm.keys():
        if not joint.startswith(pfx):
            continue
        if not (("Hand" in joint) or ("Finger" in joint)):
            continue
        p = _compute_end_site_pos(frame, joint, endsite_offsets_cm)
        if p is not None:
            end_pts[joint + "_EndSite"] = p.reshape(3)
    return end_pts


def _load_bvh_frames(bvh_file: str, fmt: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    复用 body replay 里的 loader（包含 Upper/Lower 命名映射，以及 Toe/ToeBase 映射）。
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    # 从 deploy_real/replay_bvh_body_to_redis.py 里复用 loader
    from deploy_real.replay_bvh_body_to_redis import _load_bvh_frames_via_gmr  # type: ignore

    return _load_bvh_frames_via_gmr(bvh_file, fmt)


def _get_pos(frame: Dict[str, Any], name: str) -> np.ndarray:
    v = frame.get(name, None)
    if isinstance(v, (list, tuple)) and len(v) >= 1:
        return np.asarray(v[0], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def _get_quat_wxyz(frame: Dict[str, Any], name: str) -> np.ndarray:
    v = frame.get(name, None)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return np.asarray(v[1], dtype=np.float32).reshape(4)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _parse_bvh_endsite_offsets_cm(bvh_path: str) -> Dict[str, np.ndarray]:
    """
    解析 BVH HIERARCHY（到 MOTION 前），提取每个关节对应的 End Site OFFSET（单位：cm，BVH 坐标系）。
    返回：{joint_name: np.array([x,y,z])}
    """
    endsite_offset: Dict[str, np.ndarray] = {}
    joint_stack: List[str] = []
    in_end_site = False
    end_owner: Optional[str] = None

    with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("MOTION"):
                break

            if line.startswith("ROOT "):
                name = line.split()[1]
                joint_stack.append(name)
                continue
            if line.startswith("JOINT "):
                name = line.split()[1]
                joint_stack.append(name)
                continue

            if line.startswith("End Site"):
                in_end_site = True
                end_owner = joint_stack[-1] if joint_stack else None
                continue

            if in_end_site and line.startswith("OFFSET "):
                parts = line.split()
                if len(parts) == 4 and end_owner:
                    endsite_offset[end_owner] = np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32
                    )
                continue

            if line.startswith("}"):
                if in_end_site:
                    in_end_site = False
                    end_owner = None
                    continue
                if joint_stack:
                    joint_stack.pop()
                continue

    return endsite_offset


def _quat_rotate_vec_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q (wxyz), without scipy.
    """
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    # t = 2 * cross(q_vec, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # v' = v + qw * t + cross(q_vec, t)
    vpx = vx + qw * tx + (qy * tz - qz * ty)
    vpy = vy + qw * ty + (qz * tx - qx * tz)
    vpz = vz + qw * tz + (qx * ty - qy * tx)
    return np.array([vpx, vpy, vpz], dtype=np.float32)


def _compute_end_site_pos(frame: Dict[str, Any], joint: str, endsite_offsets_cm: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    用末端关节的全局位姿 + End Site OFFSET 计算“真正指尖”位置（与 replay_bvh_body 的 loader 坐标系对齐）。
    - frame 里的 pos/quat 已经做过 rotation_matrix/rotation_quat 的坐标变换，并且单位是 m
    - BVH End Site OFFSET 是 cm 且在 BVH 坐标系下，因此需要：
      1) cm->m
      2) 用同一个 rotation_matrix 把 offset 向量变换到 frame 坐标系
      3) 再用关节的全局 quat 旋转 offset
    """
    if joint not in endsite_offsets_cm:
        return None
    if joint not in frame:
        return None
    off_cm = endsite_offsets_cm[joint]
    jpos = _get_pos(frame, joint)
    jq = _get_quat_wxyz(frame, joint)

    rot_m = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    off_m = (off_cm / 100.0) @ rot_m.T
    delta = _quat_rotate_vec_wxyz(jq, off_m)
    return jpos + delta


def _make_joint(pos: np.ndarray) -> list:
    # [[x,y,z], [qw,qx,qy,qz]] 其中 quat 不参与 Wuji retarget，给单位四元数占位即可
    return [pos.reshape(-1).tolist(), [1.0, 0.0, 0.0, 0.0]]


def _bvh_hand_to_tracking26(frame: Dict[str, Any], side: str, endsite_offsets_cm: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """
    把 BVH 的手部骨骼位置映射到 Wuji server 需要的 26D 字典 key。

    BVH 侧常见命名（你的文件）：
      LeftHand, LeftThumbFinger, LeftThumbFinger1, LeftThumbFinger2,
      LeftIndexFinger, LeftIndexFinger1, LeftIndexFinger2, LeftIndexFinger3,
      LeftMiddleFinger..., LeftRingFinger..., LeftPinkyFinger...

    Wuji 输入 key（见 server_wuji_hand_redis.py 的 HAND_JOINT_NAMES_26）：
      LeftHandWrist, LeftHandPalm, LeftHandThumbMetacarpal, ...
    """
    side = side.lower()
    assert side in ["left", "right"]
    pfx = "Left" if side == "left" else "Right"
    out: Dict[str, Any] = {}

    # helper: choose first existing name
    def pick(*names: str) -> str:
        for n in names:
            if n in frame:
                return n
        return names[0]

    # wrist/palm：BVH 里一般没有 Palm，用 Hand 代替即可
    hand_name = pick(f"{pfx}Hand")
    hand_pos = _get_pos(frame, hand_name)
    out[f"{pfx}HandWrist"] = _make_joint(hand_pos)
    out[f"{pfx}HandPalm"] = _make_joint(hand_pos)

    # tip helper: prefer End Site, fallback to last joint pos
    def tip_pos(last_joint: str) -> np.ndarray:
        if endsite_offsets_cm:
            p = _compute_end_site_pos(frame, last_joint, endsite_offsets_cm)
            if p is not None:
                return p
        return _get_pos(frame, last_joint)

    # thumb (3 joints in your BVH)
    out[f"{pfx}HandThumbMetacarpal"] = _make_joint(_get_pos(frame, pick(f"{pfx}ThumbFinger")))
    out[f"{pfx}HandThumbProximal"] = _make_joint(_get_pos(frame, pick(f"{pfx}ThumbFinger1", f"{pfx}ThumbFinger")))
    out[f"{pfx}HandThumbDistal"] = _make_joint(_get_pos(frame, pick(f"{pfx}ThumbFinger2", f"{pfx}ThumbFinger1")))
    out[f"{pfx}HandThumbTip"] = _make_joint(tip_pos(pick(f"{pfx}ThumbFinger2")))

    # index
    out[f"{pfx}HandIndexMetacarpal"] = _make_joint(_get_pos(frame, pick(f"{pfx}IndexFinger")))
    out[f"{pfx}HandIndexProximal"] = _make_joint(_get_pos(frame, pick(f"{pfx}IndexFinger1", f"{pfx}IndexFinger")))
    out[f"{pfx}HandIndexIntermediate"] = _make_joint(_get_pos(frame, pick(f"{pfx}IndexFinger2", f"{pfx}IndexFinger1")))
    out[f"{pfx}HandIndexDistal"] = _make_joint(_get_pos(frame, pick(f"{pfx}IndexFinger3", f"{pfx}IndexFinger2")))
    out[f"{pfx}HandIndexTip"] = _make_joint(tip_pos(pick(f"{pfx}IndexFinger3")))

    # middle
    out[f"{pfx}HandMiddleMetacarpal"] = _make_joint(_get_pos(frame, pick(f"{pfx}MiddleFinger")))
    out[f"{pfx}HandMiddleProximal"] = _make_joint(_get_pos(frame, pick(f"{pfx}MiddleFinger1", f"{pfx}MiddleFinger")))
    out[f"{pfx}HandMiddleIntermediate"] = _make_joint(_get_pos(frame, pick(f"{pfx}MiddleFinger2", f"{pfx}MiddleFinger1")))
    out[f"{pfx}HandMiddleDistal"] = _make_joint(_get_pos(frame, pick(f"{pfx}MiddleFinger3", f"{pfx}MiddleFinger2")))
    out[f"{pfx}HandMiddleTip"] = _make_joint(tip_pos(pick(f"{pfx}MiddleFinger3")))

    # ring
    out[f"{pfx}HandRingMetacarpal"] = _make_joint(_get_pos(frame, pick(f"{pfx}RingFinger")))
    out[f"{pfx}HandRingProximal"] = _make_joint(_get_pos(frame, pick(f"{pfx}RingFinger1", f"{pfx}RingFinger")))
    out[f"{pfx}HandRingIntermediate"] = _make_joint(_get_pos(frame, pick(f"{pfx}RingFinger2", f"{pfx}RingFinger1")))
    out[f"{pfx}HandRingDistal"] = _make_joint(_get_pos(frame, pick(f"{pfx}RingFinger3", f"{pfx}RingFinger2")))
    out[f"{pfx}HandRingTip"] = _make_joint(tip_pos(pick(f"{pfx}RingFinger3")))

    # little/pinky
    out[f"{pfx}HandLittleMetacarpal"] = _make_joint(_get_pos(frame, pick(f"{pfx}PinkyFinger")))
    out[f"{pfx}HandLittleProximal"] = _make_joint(_get_pos(frame, pick(f"{pfx}PinkyFinger1", f"{pfx}PinkyFinger")))
    out[f"{pfx}HandLittleIntermediate"] = _make_joint(_get_pos(frame, pick(f"{pfx}PinkyFinger2", f"{pfx}PinkyFinger1")))
    out[f"{pfx}HandLittleDistal"] = _make_joint(_get_pos(frame, pick(f"{pfx}PinkyFinger3", f"{pfx}PinkyFinger2")))
    out[f"{pfx}HandLittleTip"] = _make_joint(tip_pos(pick(f"{pfx}PinkyFinger3")))

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Replay BVH to Wuji via Redis (Wuji only)")
    p.add_argument("--bvh_file", type=str, required=True)
    p.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    p.add_argument("--redis_ip", type=str, default="localhost")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--hands", choices=["left", "right", "both"], default="both")
    p.add_argument("--print_every", type=int, default=60)
    p.add_argument("--viz", action="store_true", help="边 replay 边实时可视化（matplotlib 窗口）")
    p.add_argument("--viz_lines", action="store_true", help="可视化时画出手部骨架连线（Hand->Finger->...->EndSite）")
    p.add_argument("--viz_every", type=int, default=2, help="每 N 帧更新一次可视化（默认 2，避免卡）")
    p.add_argument("--viz_coords", choices=["world", "wrist_local"], default="wrist_local", help="可视化坐标系：world 或以手腕为原点")
    p.add_argument("--viz_scale", type=float, default=100.0, help="可视化缩放（默认 100 => m->cm）")
    p.add_argument("--viz_fixed_range", type=float, default=25.0, help="固定坐标范围 [-r,r]（缩放后单位，默认 25cm）")
    p.add_argument("--viz_layout", choices=["both", "left", "right"], default="both", help="可视化布局：双手/仅左/仅右")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    # Resolve BVH path: replay_bvh_wuji.sh 会 cd 到 deploy_real，用户常给的是仓库根目录相对路径
    bvh_path = args.bvh_file
    if not os.path.isabs(bvh_path) and not os.path.exists(bvh_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cand = os.path.join(repo_root, bvh_path)
        if os.path.exists(cand):
            bvh_path = cand
    if not os.path.exists(bvh_path):
        print(f"❌ 找不到 BVH 文件: {args.bvh_file}", file=sys.stderr)
        print("   你可以传绝对路径，或在仓库根目录运行脚本。", file=sys.stderr)
        return 2

    frames, _hh = _load_bvh_frames(bvh_path, args.format)
    # Parse End Site offsets once (for fingertip positions)
    endsite_offsets_cm = _parse_bvh_endsite_offsets_cm(bvh_path)
    if endsite_offsets_cm:
        print(f"[info] parsed end sites: {len(endsite_offsets_cm)} joints have End Site")

    n = len(frames)
    if n <= 0:
        print("❌ BVH 没有帧数据", file=sys.stderr)
        return 2
    start = max(0, int(args.start))
    end = n if int(args.end) < 0 else min(n, int(args.end))
    if start >= end:
        print(f"❌ start/end 非法: start={start}, end={end}, total={n}", file=sys.stderr)
        return 2

    client = redis.Redis(host=args.redis_ip, port=6379, db=0, decode_responses=False)
    try:
        client.ping()
    except Exception as e:
        print(f"❌ Redis 连接失败: {e}", file=sys.stderr)
        return 3

    robot_key = "unitree_g1_with_hands"
    key_tracking_l = f"hand_tracking_left_{robot_key}"
    key_tracking_r = f"hand_tracking_right_{robot_key}"
    key_mode_l = f"wuji_hand_mode_left_{robot_key}"
    key_mode_r = f"wuji_hand_mode_right_{robot_key}"

    rate = _RateLimiter(args.fps)
    print("=" * 70)
    print("BVH Replay -> Wuji via Redis (WUJI ONLY)")
    print("=" * 70)
    print(f"bvh_file: {args.bvh_file}")
    print(f"format  : {args.format}")
    print(f"frames  : {n} (play [{start}, {end}))")
    print(f"fps     : {args.fps}")
    print(f"loop    : {args.loop}")
    print(f"hands   : {args.hands}")
    print(f"redis   : {args.redis_ip}:6379")
    if args.dry_run:
        print("dry_run : True (不会写 Redis)")
    print("")

    i = start
    step = 0

    # Optional realtime visualization
    viz = None
    if args.viz:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            print(f"❌ 无法启用 --viz（缺少 matplotlib）：{e}", file=sys.stderr)
            print("   解决：pip install matplotlib", file=sys.stderr)
            return 4

        plt.ion()
        if args.viz_layout == "both":
            fig = plt.figure(figsize=(14, 7))
            ax_l = fig.add_subplot(1, 2, 1, projection="3d")
            ax_r = fig.add_subplot(1, 2, 2, projection="3d")
        else:
            fig = plt.figure(figsize=(9, 9))
            ax_l = fig.add_subplot(1, 1, 1, projection="3d")
            ax_r = None
        viz = (plt, fig, ax_l, ax_r)

        print(f"[viz] enabled (every={args.viz_every}, coords={args.viz_coords}, scale={args.viz_scale}, range={args.viz_fixed_range}, layout={args.viz_layout})")
    try:
        while True:
            frame = frames[i]
            ts = _now_ms()

            do_left = args.hands in ["left", "both"]
            do_right = args.hands in ["right", "both"]

            pipe = client.pipeline()

            # left
            if do_left:
                left_dict = _bvh_hand_to_tracking26(frame, "left", endsite_offsets_cm=endsite_offsets_cm)
                payload_left = {"is_active": True, "timestamp": ts, **left_dict}
                pipe.set(key_mode_l, "follow")
                pipe.set(key_tracking_l, json.dumps(payload_left))
            else:
                pipe.set(key_mode_l, "default")
                pipe.set(key_tracking_l, json.dumps({"is_active": False, "timestamp": ts}))

            # right
            if do_right:
                right_dict = _bvh_hand_to_tracking26(frame, "right", endsite_offsets_cm=endsite_offsets_cm)
                payload_right = {"is_active": True, "timestamp": ts, **right_dict}
                pipe.set(key_mode_r, "follow")
                pipe.set(key_tracking_r, json.dumps(payload_right))
            else:
                pipe.set(key_mode_r, "default")
                pipe.set(key_tracking_r, json.dumps({"is_active": False, "timestamp": ts}))

            if not args.dry_run:
                pipe.execute()

            # realtime visualization (sync with replay frame index)
            if viz is not None and (step % max(1, int(args.viz_every)) == 0):
                plt, fig, ax_l, ax_r = viz
                scale = float(args.viz_scale)
                rfix = float(args.viz_fixed_range)

                def _plot(ax, side: str, title: str, color: str):
                    pts = _collect_bvh_hand_joint_points(frame, side)
                    end_pts = _collect_bvh_hand_end_sites(frame, side, endsite_offsets_cm) if endsite_offsets_cm else {}
                    if args.viz_coords == "wrist_local":
                        origin = _get_pos(frame, ("LeftHand" if side == "left" else "RightHand")) if (("LeftHand" if side == "left" else "RightHand") in frame) else None
                    else:
                        origin = None
                    pts_t = _transform_points(pts, origin, scale)
                    end_t = _transform_points(end_pts, origin, scale)

                    ax.cla()
                    if pts_t:
                        P = np.stack(list(pts_t.values()), axis=0)
                        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=18, c=color, depthshade=True)
                        if args.viz_lines:
                            pfx = "Left" if side == "left" else "Right"
                            edges = [
                                (f"{pfx}Hand", f"{pfx}ThumbFinger"),
                                (f"{pfx}ThumbFinger", f"{pfx}ThumbFinger1"),
                                (f"{pfx}ThumbFinger1", f"{pfx}ThumbFinger2"),
                                (f"{pfx}Hand", f"{pfx}IndexFinger"),
                                (f"{pfx}IndexFinger", f"{pfx}IndexFinger1"),
                                (f"{pfx}IndexFinger1", f"{pfx}IndexFinger2"),
                                (f"{pfx}IndexFinger2", f"{pfx}IndexFinger3"),
                                (f"{pfx}Hand", f"{pfx}MiddleFinger"),
                                (f"{pfx}MiddleFinger", f"{pfx}MiddleFinger1"),
                                (f"{pfx}MiddleFinger1", f"{pfx}MiddleFinger2"),
                                (f"{pfx}MiddleFinger2", f"{pfx}MiddleFinger3"),
                                (f"{pfx}Hand", f"{pfx}RingFinger"),
                                (f"{pfx}RingFinger", f"{pfx}RingFinger1"),
                                (f"{pfx}RingFinger1", f"{pfx}RingFinger2"),
                                (f"{pfx}RingFinger2", f"{pfx}RingFinger3"),
                                (f"{pfx}Hand", f"{pfx}PinkyFinger"),
                                (f"{pfx}PinkyFinger", f"{pfx}PinkyFinger1"),
                                (f"{pfx}PinkyFinger1", f"{pfx}PinkyFinger2"),
                                (f"{pfx}PinkyFinger2", f"{pfx}PinkyFinger3"),
                            ]
                            # joint->joint lines
                            for a, b in edges:
                                if a in pts_t and b in pts_t:
                                    pa, pb = pts_t[a], pts_t[b]
                                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], c=color, linewidth=2.0, alpha=0.85)
                    if end_t:
                        P = np.stack(list(end_t.values()), axis=0)
                        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=18, c="tab:red", marker="x")
                        for name, p in end_t.items():
                            j = name.replace("_EndSite", "")
                            if j in pts_t:
                                jp = pts_t[j]
                                ax.plot([jp[0], p[0]], [jp[1], p[1]], [jp[2], p[2]], c="tab:red", linewidth=1.5, alpha=0.9)

                    ax.set_title(title)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                    _apply_fixed_range(ax, rfix)

                if args.viz_layout == "both":
                    _plot(ax_l, "left", f"Left frame={i}", "tab:blue")
                    assert ax_r is not None
                    _plot(ax_r, "right", f"Right frame={i}", "tab:orange")
                    fig.suptitle("BVH -> Wuji replay (realtime)", fontsize=12)
                elif args.viz_layout == "left":
                    _plot(ax_l, "left", f"Left frame={i}", "tab:blue")
                    fig.suptitle("BVH -> Wuji replay (left)", fontsize=12)
                else:
                    _plot(ax_l, "right", f"Right frame={i}", "tab:orange")
                    fig.suptitle("BVH -> Wuji replay (right)", fontsize=12)

                fig.canvas.draw_idle()
                plt.pause(0.001)

            if args.print_every > 0 and (step % int(args.print_every) == 0):
                print(f"[wuji-replay] frame={i} step={step} ts_ms={ts}")

            i += 1
            step += 1
            if i >= end:
                if args.loop:
                    i = start
                else:
                    break
            rate.sleep()
    finally:
        # 退出时把手置回 default，避免停在奇怪姿态
        if not args.dry_run:
            ts = _now_ms()
            pipe = client.pipeline()
            pipe.set(key_mode_l, "default")
            pipe.set(key_mode_r, "default")
            pipe.set(key_tracking_l, json.dumps({"is_active": False, "timestamp": ts}))
            pipe.set(key_tracking_r, json.dumps({"is_active": False, "timestamp": ts}))
            pipe.execute()
        try:
            if viz is not None:
                plt, _fig, _ax_l, _ax_r = viz
                plt.ioff()
        except Exception:
            pass

    print("✅ wuji replay finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


