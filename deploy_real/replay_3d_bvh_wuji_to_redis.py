#!/usr/bin/env python3
"""
Replay 3D FK pose CSV to Wuji hands via Redis (WUJI ONLY).

做什么：
- 读取 FK 后的 pose CSV（lhand_/rhand_ 的 px/py/pz + qw/qx/qy/qz）
- 从手部骨骼（LeftHand/LeftIndexFinger...）构造 Wuji server 需要的 26D hand_tracking 字典：
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


def _safe_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Quaternion (wxyz) -> 3x3 rotation matrix.
    Mirror vendor MATLAB `T_SY.m`.
    """
    q = _safe_quat_wxyz(q)
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [qw * qw + qx * qx - qy * qy - qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qw * qy + 2 * qx * qz],
            [2 * qw * qz + 2 * qx * qy, qw * qw - qx * qx + qy * qy - qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qw * qx + 2 * qy * qz, qw * qw - qx * qx - qy * qy + qz * qz],
        ],
        dtype=np.float32,
    )


def _hand_joint_order_names(prefix: str) -> List[str]:
    p = str(prefix)
    return [
        f"{p}Hand",
        f"{p}ThumbFinger",
        f"{p}ThumbFinger1",
        f"{p}ThumbFinger2",
        f"{p}IndexFinger",
        f"{p}IndexFinger1",
        f"{p}IndexFinger2",
        f"{p}IndexFinger3",
        f"{p}MiddleFinger",
        f"{p}MiddleFinger1",
        f"{p}MiddleFinger2",
        f"{p}MiddleFinger3",
        f"{p}RingFinger",
        f"{p}RingFinger1",
        f"{p}RingFinger2",
        f"{p}RingFinger3",
        f"{p}PinkyFinger",
        f"{p}PinkyFinger1",
        f"{p}PinkyFinger2",
        f"{p}PinkyFinger3",
    ]


def _fk_hand_positions_with_end_sites(
    quats_wxyz: np.ndarray,
    *,
    root_pos: np.ndarray,
    bone_init_pos: np.ndarray,
    end_site_scale: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Port of vendor MATLAB Quat2PositionLHand/Quat2PositionRHand (+ EndSite).
    """
    q = np.asarray(quats_wxyz, dtype=np.float32).reshape(20, 4)
    root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3)
    bone = np.asarray(bone_init_pos, dtype=np.float32).reshape(20, 3)

    parent = np.array([0, 0, 1, 2, 0, 4, 5, 6, 0, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18], dtype=np.int32)
    bone_ver = bone - bone[parent]

    pos = np.zeros((20, 3), dtype=np.float32)
    pos[0] = root_pos
    for j in range(1, 20):
        pidx = int(parent[j])
        R = _quat_wxyz_to_rotmat(q[pidx])
        pos[j] = pos[pidx] + (R @ bone_ver[j].reshape(3, 1)).reshape(3)

    end_joint = np.array([3, 7, 11, 15, 19], dtype=np.int32)
    prev_joint = np.array([2, 6, 10, 14, 18], dtype=np.int32)
    bone_ver_end = (bone[end_joint] - bone[prev_joint]) * float(end_site_scale)

    pos_end = np.zeros((5, 3), dtype=np.float32)
    for k in range(5):
        j = int(end_joint[k])
        R = _quat_wxyz_to_rotmat(q[j])
        pos_end[k] = pos[j] + (R @ bone_ver_end[k].reshape(3, 1)).reshape(3)
    return pos, pos_end


# Default T-pose hand skeleton (same as SDK init; units: meters)
INITIAL_POSITION_HAND_RIGHT = [
    [0.682, -0.061, 1.526],
    [0.71, -0.024, 1.526],
    [0.728, -0.008, 1.526],
    [0.755, 0.013, 1.526],
    [0.707, -0.05, 1.526],
    [0.761, -0.024, 1.525],
    [0.812, -0.023, 1.525],
    [0.837, -0.022, 1.525],
    [0.709, -0.058, 1.526],
    [0.764, -0.046, 1.528],
    [0.816, -0.046, 1.528],
    [0.845, -0.046, 1.528],
    [0.709, -0.064, 1.526],
    [0.761, -0.069, 1.527],
    [0.812, -0.069, 1.527],
    [0.835, -0.069, 1.527],
    [0.708, -0.072, 1.526],
    [0.755, -0.089, 1.522],
    [0.791, -0.089, 1.522],
    [0.81, -0.089, 1.522],
]

INITIAL_POSITION_HAND_LEFT = [
    [-0.682, -0.061, 1.526],
    [-0.71, -0.024, 1.526],
    [-0.728, -0.008, 1.526],
    [-0.755, 0.013, 1.526],
    [-0.707, -0.05, 1.526],
    [-0.761, -0.024, 1.525],
    [-0.812, -0.023, 1.525],
    [-0.837, -0.022, 1.525],
    [-0.709, -0.058, 1.526],
    [-0.764, -0.046, 1.528],
    [-0.816, -0.046, 1.528],
    [-0.845, -0.046, 1.528],
    [-0.709, -0.064, 1.526],
    [-0.761, -0.069, 1.527],
    [-0.812, -0.069, 1.527],
    [-0.835, -0.069, 1.527],
    [-0.708, -0.072, 1.526],
    [-0.755, -0.089, 1.522],
    [-0.791, -0.089, 1.522],
    [-0.81, -0.089, 1.522],
]


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


def _load_pose_csv_frames(csv_file: str) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """
    Load pose CSV and return frames containing only hand joints (Left*/Right* from lhand_/rhand_).
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from deploy_real.pose_csv_loader import load_pose_csv_frames  # type: ignore

    frames, meta = load_pose_csv_frames(
        csv_file, include_body=False, include_lhand=True, include_rhand=True, max_frames=-1
    )
    return frames, meta.fps


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


def _parse_bvh_parent_map(bvh_path: str) -> Dict[str, Optional[str]]:
    """
    解析 BVH HIERARCHY（到 MOTION 前），提取 parent 关系：{joint: parent_joint}
    只用于可视化连线。
    """
    parent: Dict[str, Optional[str]] = {}
    joint_stack: List[str] = []
    in_end_site = False

    with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("MOTION"):
                break

            if line.startswith("ROOT "):
                name = line.split()[1]
                parent[name] = None
                joint_stack.append(name)
                continue

            if line.startswith("JOINT "):
                name = line.split()[1]
                pj = joint_stack[-1] if joint_stack else None
                parent[name] = pj
                joint_stack.append(name)
                continue

            if line.startswith("End Site"):
                in_end_site = True
                continue

            if line.startswith("}"):
                if in_end_site:
                    in_end_site = False
                    continue
                if joint_stack:
                    joint_stack.pop()
                continue

    return parent


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


def _bvh_hand_to_tracking26(
    frame: Dict[str, Any],
    side: str,
    endsite_offsets_cm: Optional[Dict[str, np.ndarray]] = None,
    *,
    pos_override: Optional[Dict[str, np.ndarray]] = None,
    tip_override: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
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

    def get_pos(name: str) -> np.ndarray:
        if pos_override is not None and name in pos_override:
            return np.asarray(pos_override[name], dtype=np.float32).reshape(3)
        return _get_pos(frame, name).reshape(3)

    # wrist/palm：BVH 里一般没有 Palm，用 Hand 代替即可
    hand_name = pick(f"{pfx}Hand")
    hand_pos = get_pos(hand_name)
    out[f"{pfx}HandWrist"] = _make_joint(hand_pos)
    out[f"{pfx}HandPalm"] = _make_joint(hand_pos)

    # tip helper: prefer End Site, fallback to last joint pos
    def tip_pos(last_joint: str) -> np.ndarray:
        if tip_override is not None:
            key_map = {
                f"{pfx}ThumbFinger2": f"{pfx}HandThumbTip",
                f"{pfx}IndexFinger3": f"{pfx}HandIndexTip",
                f"{pfx}MiddleFinger3": f"{pfx}HandMiddleTip",
                f"{pfx}RingFinger3": f"{pfx}HandRingTip",
                f"{pfx}PinkyFinger3": f"{pfx}HandLittleTip",
            }
            k = key_map.get(last_joint, "")
            if k and (k in tip_override):
                return np.asarray(tip_override[k], dtype=np.float32).reshape(3)
        if endsite_offsets_cm:
            p = _compute_end_site_pos(frame, last_joint, endsite_offsets_cm)
            if p is not None:
                return p
        return get_pos(last_joint)

    # thumb (3 joints in your BVH)
    out[f"{pfx}HandThumbMetacarpal"] = _make_joint(get_pos(pick(f"{pfx}ThumbFinger")))
    out[f"{pfx}HandThumbProximal"] = _make_joint(get_pos(pick(f"{pfx}ThumbFinger1", f"{pfx}ThumbFinger")))
    out[f"{pfx}HandThumbDistal"] = _make_joint(get_pos(pick(f"{pfx}ThumbFinger2", f"{pfx}ThumbFinger1")))
    out[f"{pfx}HandThumbTip"] = _make_joint(tip_pos(pick(f"{pfx}ThumbFinger2")))

    # index
    out[f"{pfx}HandIndexMetacarpal"] = _make_joint(get_pos(pick(f"{pfx}IndexFinger")))
    out[f"{pfx}HandIndexProximal"] = _make_joint(get_pos(pick(f"{pfx}IndexFinger1", f"{pfx}IndexFinger")))
    out[f"{pfx}HandIndexIntermediate"] = _make_joint(get_pos(pick(f"{pfx}IndexFinger2", f"{pfx}IndexFinger1")))
    out[f"{pfx}HandIndexDistal"] = _make_joint(get_pos(pick(f"{pfx}IndexFinger3", f"{pfx}IndexFinger2")))
    out[f"{pfx}HandIndexTip"] = _make_joint(tip_pos(pick(f"{pfx}IndexFinger3")))

    # middle
    out[f"{pfx}HandMiddleMetacarpal"] = _make_joint(get_pos(pick(f"{pfx}MiddleFinger")))
    out[f"{pfx}HandMiddleProximal"] = _make_joint(get_pos(pick(f"{pfx}MiddleFinger1", f"{pfx}MiddleFinger")))
    out[f"{pfx}HandMiddleIntermediate"] = _make_joint(get_pos(pick(f"{pfx}MiddleFinger2", f"{pfx}MiddleFinger1")))
    out[f"{pfx}HandMiddleDistal"] = _make_joint(get_pos(pick(f"{pfx}MiddleFinger3", f"{pfx}MiddleFinger2")))
    out[f"{pfx}HandMiddleTip"] = _make_joint(tip_pos(pick(f"{pfx}MiddleFinger3")))

    # ring
    out[f"{pfx}HandRingMetacarpal"] = _make_joint(get_pos(pick(f"{pfx}RingFinger")))
    out[f"{pfx}HandRingProximal"] = _make_joint(get_pos(pick(f"{pfx}RingFinger1", f"{pfx}RingFinger")))
    out[f"{pfx}HandRingIntermediate"] = _make_joint(get_pos(pick(f"{pfx}RingFinger2", f"{pfx}RingFinger1")))
    out[f"{pfx}HandRingDistal"] = _make_joint(get_pos(pick(f"{pfx}RingFinger3", f"{pfx}RingFinger2")))
    out[f"{pfx}HandRingTip"] = _make_joint(tip_pos(pick(f"{pfx}RingFinger3")))

    # little/pinky
    out[f"{pfx}HandLittleMetacarpal"] = _make_joint(get_pos(pick(f"{pfx}PinkyFinger")))
    out[f"{pfx}HandLittleProximal"] = _make_joint(get_pos(pick(f"{pfx}PinkyFinger1", f"{pfx}PinkyFinger")))
    out[f"{pfx}HandLittleIntermediate"] = _make_joint(get_pos(pick(f"{pfx}PinkyFinger2", f"{pfx}PinkyFinger1")))
    out[f"{pfx}HandLittleDistal"] = _make_joint(get_pos(pick(f"{pfx}PinkyFinger3", f"{pfx}PinkyFinger2")))
    out[f"{pfx}HandLittleTip"] = _make_joint(tip_pos(pick(f"{pfx}PinkyFinger3")))

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Replay 3D pose CSV to Wuji via Redis (Wuji only)")
    p.add_argument("--csv_file", type=str, required=True)
    p.add_argument("--redis_ip", type=str, default="localhost")
    p.add_argument("--fps", type=float, default=30.0, help="Replay FPS. If --use_csv_fps, overrides from CSV frequency")
    p.add_argument("--use_csv_fps", action="store_true", help="Use FPS from CSV 'frequency' column if available")
    p.add_argument("--csv_calib_json", type=str, default="", help="Apply a pre-derived CSV calibration json (so replay does NOT need a BVH file). This is applied before other CSV transforms.")
    p.add_argument("--csv_units", choices=["m", "cm", "mm"], default="m", help="Units of CSV position fields (default: m)")
    p.add_argument("--csv_apply_bvh_rotation", action="store_true", help="Apply BVH-like axis rotation to CSV pos/quat (convert world coords to GMR convention)")
    p.add_argument("--csv_geo_to_bvh_official", action="store_true", help="Apply vendor official Geo->BVH mapping to CSV pos+quat: pos(-x,z,y), quat(w,-x,z,y). Intended for BVH raw world.")
    p.add_argument("--csv_geo_to_bvh_official_pos_only", action="store_true", help="Apply vendor official Geo->BVH mapping to CSV positions only: pos(-x,z,y), keep quats unchanged.")
    p.add_argument("--csv_quat_order", choices=["wxyz", "xyzw"], default="wxyz", help="Quaternion order in CSV (default: wxyz)")
    p.add_argument("--csv_quat_space", choices=["global", "local"], default="global", help="Quaternion space in CSV (default: global)")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--hands", choices=["left", "right", "both"], default="both")
    p.add_argument("--print_every", type=int, default=60)
    p.add_argument("--viz", action="store_true", help="边 replay 边实时可视化（matplotlib 窗口）")
    p.add_argument("--viz_every", type=int, default=2, help="每 N 帧更新一次可视化（默认 2，避免卡）")
    p.add_argument("--viz_coords", choices=["world", "wrist_local"], default="wrist_local", help="可视化坐标系：world 或以手腕为原点")
    p.add_argument("--viz_scale", type=float, default=100.0, help="可视化缩放（默认 100 => m->cm）")
    p.add_argument("--viz_fixed_range", type=float, default=25.0, help="固定坐标范围 [-r,r]（缩放后单位，默认 25cm）")
    p.add_argument("--viz_layout", choices=["both", "left", "right"], default="both", help="可视化布局：双手/仅左/仅右")
    p.add_argument("--hand_fk", action="store_true", help="Use vendor FK (quats + T-pose offsets) to compute joint positions + fingertip EndSites")
    p.add_argument("--hand_fk_end_site_scale", type=float, default=0.8, help="EndSite length scale (vendor MATLAB default=0.8)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    # FK currently supports the 'official transform' path; avoid mixing with csv_calib_json for now.
    if bool(getattr(args, "hand_fk", False)) and bool(getattr(args, "csv_calib_json", "")):
        print("⚠️ 设置了 --hand_fk 但同时用了 --csv_calib_json：为避免坐标系/校准重复，这里将禁用 hand_fk", file=sys.stderr)
        setattr(args, "hand_fk", False)

    # Resolve CSV path
    csv_path = args.csv_file
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cand = os.path.join(repo_root, csv_path)
        if os.path.exists(cand):
            csv_path = cand
    if not os.path.exists(csv_path):
        print(f"❌ 找不到 CSV 文件: {args.csv_file}", file=sys.stderr)
        print("   你可以传绝对路径，或在仓库根目录运行脚本。", file=sys.stderr)
        return 2

    frames, fps_from_csv = _load_pose_csv_frames(csv_path)
    if bool(getattr(args, "csv_calib_json", "")) or args.csv_quat_order != "wxyz" or args.csv_quat_space != "global" or bool(getattr(args, "csv_geo_to_bvh_official", False)) or bool(getattr(args, "csv_geo_to_bvh_official_pos_only", False)) or args.csv_apply_bvh_rotation or args.csv_units != "m":
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from deploy_real.pose_csv_loader import (
            apply_bvh_like_coordinate_transform,
            convert_quat_order,
            default_parent_map_body,
            quats_local_to_global,
            apply_geo_to_bvh_official,
            apply_geo_to_bvh_official_pos_only,
            load_csv_calib_json,
            apply_csv_calib_to_frames,
        )  # type: ignore

        calib_path = str(getattr(args, "csv_calib_json", "")).strip()
        if calib_path:
            if not os.path.isabs(calib_path) and not os.path.exists(calib_path):
                cand = os.path.join(repo_root, calib_path)
                if os.path.exists(cand):
                    calib_path = cand
            if not os.path.exists(calib_path):
                print(f"❌ 找不到 csv_calib_json: {getattr(args, 'csv_calib_json', '')}", file=sys.stderr)
                return 2

            if args.csv_apply_bvh_rotation or args.csv_geo_to_bvh_official or args.csv_geo_to_bvh_official_pos_only:
                print("❌ 使用 --csv_calib_json 时，请不要再叠加 --csv_apply_bvh_rotation/--csv_geo_to_bvh_official*（避免重复坐标变换）", file=sys.stderr)
                return 2
            if args.csv_units != "m":
                print("❌ 使用 --csv_calib_json 时，请不要再设置 --csv_units（单位由 calib.json 决定）", file=sys.stderr)
                return 2
            if args.csv_quat_order != "wxyz" or args.csv_quat_space != "global":
                print("❌ 使用 --csv_calib_json 时，请不要再设置 quat order/space（由 calib.json 决定）", file=sys.stderr)
                return 2

            calib = load_csv_calib_json(calib_path)
            # For Wuji, we only use positions, but applying the same calibration ensures the same world convention.
            frames = apply_csv_calib_to_frames(frames, calib, fmt=None, recompute_footmod=False)

        if args.csv_quat_order != "wxyz":
            frames = [convert_quat_order(fr, args.csv_quat_order) for fr in frames]
        if args.csv_quat_space == "local":
            # For CSV, hands are global positions already; if quats are local, we can at least globalize body chain.
            pm = default_parent_map_body()
            frames = [quats_local_to_global(fr, pm) for fr in frames]

        if bool(getattr(args, "csv_geo_to_bvh_official", False)) or bool(getattr(args, "csv_geo_to_bvh_official_pos_only", False)):
            if bool(getattr(args, "csv_geo_to_bvh_official", False)) and bool(getattr(args, "csv_geo_to_bvh_official_pos_only", False)):
                print("⚠️ 同时设置了 --csv_geo_to_bvh_official 与 --csv_geo_to_bvh_official_pos_only，将优先使用 pos+quat 版本", file=sys.stderr)
            if bool(getattr(args, "csv_geo_to_bvh_official", False)):
                frames = [apply_geo_to_bvh_official(fr) for fr in frames]
            else:
                frames = [apply_geo_to_bvh_official_pos_only(fr) for fr in frames]
        if args.csv_apply_bvh_rotation or args.csv_units != "m":
            frames = [
                apply_bvh_like_coordinate_transform(fr, pos_unit=args.csv_units, apply_rotation=bool(args.csv_apply_bvh_rotation))
                for fr in frames
            ]
    # NOTE: CSV 没有 End Site：tip 直接用 Finger3 / ThumbFinger2
    endsite_offsets_cm: Dict[str, np.ndarray] = {}

    # Precompute transformed T-pose bones for vendor FK, consistent with frame transforms.
    fk_bone_left: Optional[np.ndarray] = None
    fk_bone_right: Optional[np.ndarray] = None
    if bool(getattr(args, "hand_fk", False)):
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from deploy_real.pose_csv_loader import (  # type: ignore
                apply_bvh_like_coordinate_transform as _apply_bvh_like_coordinate_transform,
                apply_geo_to_bvh_official as _apply_geo_to_bvh_official,
                apply_geo_to_bvh_official_pos_only as _apply_geo_to_bvh_official_pos_only,
            )

            def _transform_init_bone(bone_xyz_list: List[List[float]], prefix: str) -> np.ndarray:
                names = _hand_joint_order_names(prefix)
                fr: Dict[str, Any] = {}
                for n, p0 in zip(names, bone_xyz_list):
                    fr[n] = [np.asarray(p0, dtype=np.float32).reshape(3), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]
                if bool(getattr(args, "csv_geo_to_bvh_official", False)):
                    fr = _apply_geo_to_bvh_official(fr)
                elif bool(getattr(args, "csv_geo_to_bvh_official_pos_only", False)):
                    fr = _apply_geo_to_bvh_official_pos_only(fr)
                if bool(getattr(args, "csv_apply_bvh_rotation", False)):
                    fr = _apply_bvh_like_coordinate_transform(fr, pos_unit="m", apply_rotation=True)
                return np.stack([np.asarray(fr[n][0], dtype=np.float32).reshape(3) for n in names], axis=0)

            fk_bone_left = _transform_init_bone(INITIAL_POSITION_HAND_LEFT, "Left")
            fk_bone_right = _transform_init_bone(INITIAL_POSITION_HAND_RIGHT, "Right")
        except Exception as e:
            print(f"⚠️ hand_fk 初始化失败，将回退不用 FK：{e}", file=sys.stderr)
            fk_bone_left = None
            fk_bone_right = None

    n = len(frames)
    if n <= 0:
        print("❌ CSV 没有帧数据", file=sys.stderr)
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

    fps = float(args.fps)
    if args.use_csv_fps and fps_from_csv and fps_from_csv > 1e-6:
        fps = float(fps_from_csv)
        print(f"[info] use_csv_fps=True, override fps => {fps}")
    rate = _RateLimiter(fps)
    print("=" * 70)
    print("3D Pose CSV Replay -> Wuji via Redis (WUJI ONLY)")
    print("=" * 70)
    print(f"csv_file: {args.csv_file}")
    print(f"frames  : {n} (play [{start}, {end}))")
    print(f"fps     : {fps}")
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
                pos_override = None
                tip_override = None
                if bool(getattr(args, "hand_fk", False)) and (fk_bone_left is not None):
                    try:
                        prefix = "Left"
                        names = _hand_joint_order_names(prefix)
                        q20 = np.stack([_safe_quat_wxyz(_get_quat_wxyz(frame, n)) for n in names], axis=0)
                        root_pos = _get_pos(frame, f"{prefix}Hand").reshape(3)
                        pos20, pos_end5 = _fk_hand_positions_with_end_sites(
                            q20,
                            root_pos=root_pos,
                            bone_init_pos=fk_bone_left,
                            end_site_scale=float(getattr(args, "hand_fk_end_site_scale", 0.8)),
                        )
                        pos_override = {n: pos20[i] for i, n in enumerate(names)}
                        tip_override = {
                            f"{prefix}HandThumbTip": pos_end5[0],
                            f"{prefix}HandIndexTip": pos_end5[1],
                            f"{prefix}HandMiddleTip": pos_end5[2],
                            f"{prefix}HandRingTip": pos_end5[3],
                            f"{prefix}HandLittleTip": pos_end5[4],
                        }
                    except Exception:
                        pos_override = None
                        tip_override = None
                left_dict = _bvh_hand_to_tracking26(frame, "left", endsite_offsets_cm=None, pos_override=pos_override, tip_override=tip_override)
                payload_left = {"is_active": True, "timestamp": ts, **left_dict}
                pipe.set(key_mode_l, "follow")
                pipe.set(key_tracking_l, json.dumps(payload_left))
            else:
                pipe.set(key_mode_l, "default")
                pipe.set(key_tracking_l, json.dumps({"is_active": False, "timestamp": ts}))

            # right
            if do_right:
                pos_override = None
                tip_override = None
                if bool(getattr(args, "hand_fk", False)) and (fk_bone_right is not None):
                    try:
                        prefix = "Right"
                        names = _hand_joint_order_names(prefix)
                        q20 = np.stack([_safe_quat_wxyz(_get_quat_wxyz(frame, n)) for n in names], axis=0)
                        root_pos = _get_pos(frame, f"{prefix}Hand").reshape(3)
                        pos20, pos_end5 = _fk_hand_positions_with_end_sites(
                            q20,
                            root_pos=root_pos,
                            bone_init_pos=fk_bone_right,
                            end_site_scale=float(getattr(args, "hand_fk_end_site_scale", 0.8)),
                        )
                        pos_override = {n: pos20[i] for i, n in enumerate(names)}
                        tip_override = {
                            f"{prefix}HandThumbTip": pos_end5[0],
                            f"{prefix}HandIndexTip": pos_end5[1],
                            f"{prefix}HandMiddleTip": pos_end5[2],
                            f"{prefix}HandRingTip": pos_end5[3],
                            f"{prefix}HandLittleTip": pos_end5[4],
                        }
                    except Exception:
                        pos_override = None
                        tip_override = None
                right_dict = _bvh_hand_to_tracking26(frame, "right", endsite_offsets_cm=None, pos_override=pos_override, tip_override=tip_override)
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


