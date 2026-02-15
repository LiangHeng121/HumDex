#!/usr/bin/env python3
"""
XDMocap / VMC 实时遥操（BODY ONLY）-> Redis

数据链路（与 replay_3d_bvh.sh 的“坐标/四元数变换”保持一致）：
  XDMocap SDK UDP (WS_Geo, pos=xyz(m), quat=wxyz global)
    --(官方 Geo->BVH: pos(-x,z,y), quat(w,-x,z,y))--> BVH raw world
    --(BVH->GMR 固定轴旋转)--> GMR world
    --(GMR IK retarget)--> action_body_* (35D mimic obs) -> Redis

说明：
- 写 action_body_*（35D）+ hand_tracking_*（Wuji follow）
- action_hand_*/action_neck_* 不控制（固定默认）
- 支持键盘开关（与 teleop.sh 一致）：
  - k: 切换 send_enabled（禁用时发送安全 idle action，并把 Wuji 置 default）
  - p: 切换 hold_position（冻结动作，重复上一帧 action；Wuji 置 hold）
 - body_source=vmc 时，body 通过 VMC/OSC 接收（例如 SlimeVR）
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import subprocess
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np  # type: ignore
try:
    import redis  # type: ignore
except Exception as e:
    raise SystemExit(
        "❌ 缺少依赖 `redis`（python-redis）。\n"
        "   解决：在你的运行环境里安装，例如：\n"
        "     - pip install redis\n"
        "     - 或 conda install -c conda-forge redis-py\n"
        f"   原始错误：{e}"
    )
from loop_rate_limiters import RateLimiter  # type: ignore
from rich import print  # type: ignore
from data_utils.evdev_hotkeys import EvdevHotkeys, EvdevHotkeyConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def _parse_safe_idle_pose_ids(arg: Any) -> list[int]:
    """
    解析 --safe_idle_pose_id。
    支持：
      - 单个数字：2
      - 逗号分隔列表："1,2"
    """
    if arg is None:
        return [0]
    if isinstance(arg, int):
        ids = [int(arg)]
    else:
        s = str(arg).strip()
        if s == "":
            ids = [0]
        else:
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            if not parts:
                ids = [0]
            else:
                ids = []
                for p in parts:
                    try:
                        ids.append(int(p))
                    except Exception as e:
                        raise ValueError(f"--safe_idle_pose_id 解析失败：{arg!r}（无法解析 {p!r} 为整数）") from e

    missing = [i for i in ids if i not in SAFE_IDLE_BODY_35_PRESETS]
    if missing:
        raise ValueError(
            f"--safe_idle_pose_id 包含不存在的 preset id：{missing}。"
            f"可用范围：{sorted(SAFE_IDLE_BODY_35_PRESETS.keys())}"
        )
    return ids


# ---------------------------------------------------------------------
# VMC (OSC) receiver for SlimeVR body input
# ---------------------------------------------------------------------
def _normalize_vmc_name(name: str) -> str:
    return "".join([c.lower() for c in str(name) if c.isalnum()])


def _vmc_quat_xyzw_to_wxyz(qx: float, qy: float, qz: float, qw: float, invert_zw: bool) -> np.ndarray:
    if invert_zw:
        qz = -float(qz)
        qw = -float(qw)
    return _safe_quat_wxyz(np.array([qw, qx, qy, qz], dtype=np.float32))


def _quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
    q = _safe_quat_wxyz(np.asarray(q, dtype=np.float32).reshape(4))
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _quat_to_mat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        q = (q / n).astype(np.float32)
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _mat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32).reshape(3, 3)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _safe_quat_wxyz(np.array([w, x, y, z], dtype=np.float32))


def _axis_swap_flip_matrix(swap: str, flip: str) -> np.ndarray:
    swap = str(swap).lower()
    flip = str(flip).lower()
    axes = {"x": 0, "y": 1, "z": 2}
    if len(swap) != 3 or any(c not in axes for c in swap):
        swap = "xyz"
    idx = [axes[c] for c in swap]
    m = np.eye(3, dtype=np.float32)[:, idx]
    f = np.ones(3, dtype=np.float32)
    for c in flip:
        if c in axes:
            f[axes[c]] *= -1.0
    return m * f.reshape(1, 3)


def _parse_bvh_offsets(path: str) -> tuple[dict[str, Optional[str]], dict[str, np.ndarray]]:
    parents: dict[str, Optional[str]] = {}
    offsets: dict[str, np.ndarray] = {}
    stack: list[Optional[str]] = []
    current: Optional[str] = None
    in_end_site = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            head = tokens[0]
            if head in ["ROOT", "JOINT"]:
                name = tokens[1]
                current = name
                parent = stack[-1] if stack else None
                parents[name] = parent
                stack.append(name)
            elif head == "End" and len(tokens) >= 2 and tokens[1] == "Site":
                in_end_site = True
                stack.append(None)
            elif head == "OFFSET" and len(tokens) >= 4:
                if not in_end_site and current is not None:
                    offsets[current] = np.array(
                        [float(tokens[1]), float(tokens[2]), float(tokens[3])], dtype=np.float32
                    )
            elif head == "}":
                if stack:
                    top = stack.pop()
                    if top is None:
                        in_end_site = False
                current = stack[-1] if stack else None
            elif head == "MOTION":
                break
    return parents, offsets


def _std_fk_skeleton() -> tuple[dict[str, Optional[str]], dict[str, np.ndarray], np.ndarray]:
    # Match deploy_real/vmc_fk_viewer.py STD_SKELETON
    offsets: dict[str, np.ndarray] = {}
    parents: dict[str, Optional[str]] = {}
    def add(name: str, parent: Optional[str], off: tuple[float, float, float]) -> None:
        parents[name] = parent
        offsets[name] = np.array(off, dtype=np.float32)

    add("Hips", None, (0.0, 0.0, 0.0))
    add("Spine", "Hips", (0.0, 0.10, 0.0))
    add("Chest", "Spine", (0.0, 0.15, 0.0))
    add("UpperChest", "Chest", (0.0, 0.15, 0.0))
    add("Neck", "UpperChest", (0.0, 0.15, 0.0))
    add("Head", "Neck", (0.0, 0.10, 0.0))

    add("LeftUpperLeg", "Hips", (-0.08, -0.05, 0.0))
    add("LeftLowerLeg", "LeftUpperLeg", (0.0, -0.42, 0.0))
    add("LeftFoot", "LeftLowerLeg", (0.0, -0.40, 0.0))

    add("RightUpperLeg", "Hips", (0.08, -0.05, 0.0))
    add("RightLowerLeg", "RightUpperLeg", (0.0, -0.42, 0.0))
    add("RightFoot", "RightLowerLeg", (0.0, -0.40, 0.0))

    add("LeftShoulder", "UpperChest", (-0.10, 0.10, 0.0))
    add("LeftUpperArm", "LeftShoulder", (-0.12, 0.0, 0.0))
    add("LeftLowerArm", "LeftUpperArm", (-0.28, 0.0, 0.0))
    add("LeftHand", "LeftLowerArm", (-0.25, 0.0, 0.0))

    add("RightShoulder", "UpperChest", (0.10, 0.10, 0.0))
    add("RightUpperArm", "RightShoulder", (0.12, 0.0, 0.0))
    add("RightLowerArm", "RightUpperArm", (0.28, 0.0, 0.0))
    add("RightHand", "RightLowerArm", (0.25, 0.0, 0.0))

    root_pos = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return parents, offsets, root_pos


def _build_fk_from_vmc(
    raw_map: dict[str, tuple[np.ndarray, np.ndarray]],
    parents: dict[str, Optional[str]],
    offsets: dict[str, np.ndarray],
    bvh_to_vmc: dict[str, str],
    scale: float,
    rot_mode: str,
    root_pos: Optional[np.ndarray],
    axis_m: Optional[np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pos_out: dict[str, np.ndarray] = {}
    rot_out: dict[str, np.ndarray] = {}

    def get_local_rot(joint: str) -> np.ndarray:
        vmc_name = bvh_to_vmc.get(joint)
        if vmc_name is None:
            return np.eye(3, dtype=np.float32)
        key = _normalize_vmc_name(vmc_name)
        v = raw_map.get(key)
        if v is None:
            return np.eye(3, dtype=np.float32)
        _p, q = v
        return _quat_to_mat_wxyz(q)

    def solve(joint: str) -> None:
        if joint in pos_out:
            return
        parent = parents.get(joint)
        if parent is None:
            local_rot = get_local_rot(joint)
            if root_pos is None:
                pos_out[joint] = np.zeros(3, dtype=np.float32)
            else:
                rp = np.asarray(root_pos, dtype=np.float32).reshape(3)
                pos_out[joint] = (axis_m @ rp) if axis_m is not None else rp
            rot_out[joint] = local_rot
            return
        solve(parent)
        parent_rot = rot_out[parent]
        local_rot = get_local_rot(joint)
        if rot_mode == "global":
            local_rot = parent_rot.T @ local_rot
        rot_out[joint] = parent_rot @ local_rot
        off = offsets.get(joint, np.zeros(3, dtype=np.float32))
        off = (axis_m @ off) if axis_m is not None else off
        pos_out[joint] = pos_out[parent] + parent_rot @ (off * float(scale))

    for j in parents.keys():
        solve(j)
    return pos_out, rot_out


def _fk_to_vmc_pose(
    fk_pos: dict[str, np.ndarray],
    fk_rot: dict[str, np.ndarray],
    bvh_to_vmc: dict[str, str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for bvh_name, vmc_name in bvh_to_vmc.items():
        p = fk_pos.get(bvh_name)
        r = fk_rot.get(bvh_name)
        if p is None or r is None:
            continue
        out[_normalize_vmc_name(vmc_name)] = (np.asarray(p, dtype=np.float32).reshape(3), _mat_to_quat_wxyz(r))
    return out


def _build_fk_from_vmc_std(
    raw_xyzw: dict[str, tuple[np.ndarray, np.ndarray]],
    rot_mode: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Exact FK logic copied from deploy_real/vmc_fk_viewer.py
    parents, offsets, root_pos = _std_fk_skeleton()

    def get_rot_xyzw(name: str) -> np.ndarray:
        key = _normalize_vmc_name(name)
        v = raw_xyzw.get(key)
        if v is None:
            return np.eye(3, dtype=np.float32)
        _p, q = v
        return _quat_to_mat_xyzw(q)

    pos: dict[str, np.ndarray] = {}
    rot: dict[str, np.ndarray] = {}

    # Root: Hips at (0,1,0)
    hips_rot = get_rot_xyzw("Hips")
    pos["Hips"] = np.asarray(root_pos, dtype=np.float32).reshape(3)
    rot["Hips"] = hips_rot

    processed = {"Hips"}
    while len(processed) < len(parents):
        for bone, parent in parents.items():
            if bone in processed:
                continue
            if parent in processed:
                parent_pos = pos[parent]
                parent_rot = rot[parent]
                off = offsets.get(bone, np.zeros(3, dtype=np.float32))
                pos[bone] = parent_pos + parent_rot @ off

                bone_rot = get_rot_xyzw(bone)
                if rot_mode == "global":
                    local_rot = parent_rot.T @ bone_rot
                    rot[bone] = parent_rot @ local_rot
                else:
                    rot[bone] = parent_rot @ bone_rot
                processed.add(bone)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in parents.keys():
        if name in pos and name in rot:
            out[_normalize_vmc_name(name)] = (
                np.asarray(pos[name], dtype=np.float32).reshape(3),
                _mat_to_quat_wxyz(rot[name]),
            )
    return out


def _frame_copy(frame: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        try:
            pos = np.asarray(v[0], dtype=np.float32).reshape(3).copy()
            quat = np.asarray(v[1], dtype=np.float32).reshape(4).copy()
            out[k] = [pos, quat]
        except Exception:
            out[k] = v
    return out


def _frame_from_pos_rot(
    pos: dict[str, np.ndarray],
    rot: dict[str, np.ndarray],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, p in pos.items():
        r = rot.get(k, None)
        if r is None:
            continue
        nk = _normalize_vmc_name(k)
        out[nk] = [np.asarray(p, dtype=np.float32).reshape(3), _mat_to_quat_wxyz(np.asarray(r, dtype=np.float32))]
    return out


def _swap_lr_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if "Left" in k:
            nk = k.replace("Left", "__TMP__").replace("Right", "Left").replace("__TMP__", "Right")
        elif "Right" in k:
            nk = k.replace("Right", "__TMP__").replace("Left", "Right").replace("__TMP__", "Left")
        else:
            nk = k
        out[nk] = v
    return out


def _mirror_frame_axes(frame: Dict[str, Any], mirror_x: bool, mirror_y: bool, mirror_z: bool) -> Dict[str, Any]:
    if not (mirror_x or mirror_y or mirror_z):
        return frame
    M = np.diag([
        -1.0 if mirror_x else 1.0,
        -1.0 if mirror_y else 1.0,
        -1.0 if mirror_z else 1.0,
    ]).astype(np.float32)
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        try:
            pos = np.asarray(v[0], dtype=np.float32).reshape(3)
            quat = np.asarray(v[1], dtype=np.float32).reshape(4)
            pos2 = (M @ pos).astype(np.float32)
            R = _quat_to_mat_wxyz(quat)
            R2 = M @ R @ M
            quat2 = _mat_to_quat_wxyz(R2)
            out[k] = [pos2, quat2]
        except Exception:
            out[k] = v
    return out


def _rotate_frame(frame: Dict[str, Any], Rw: np.ndarray) -> Dict[str, Any]:
    """Apply a proper rotation (det=+1) to pos+rot in a frame."""
    Rw = np.asarray(Rw, dtype=np.float32).reshape(3, 3)
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        try:
            pos = np.asarray(v[0], dtype=np.float32).reshape(3)
            quat = np.asarray(v[1], dtype=np.float32).reshape(4)
            pos2 = (Rw @ pos).astype(np.float32)
            R = _quat_to_mat_wxyz(quat)
            R2 = Rw @ R @ Rw.T
            quat2 = _mat_to_quat_wxyz(R2)
            out[k] = [pos2, quat2]
        except Exception:
            out[k] = v
    return out

def _rpy_deg_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), degrees."""
    r = float(np.deg2rad(float(roll)))
    p = float(np.deg2rad(float(pitch)))
    y = float(np.deg2rad(float(yaw)))
    cr, sr = float(np.cos(r)), float(np.sin(r))
    cp, sp = float(np.cos(p)), float(np.sin(p))
    cy, sy = float(np.cos(y)), float(np.sin(y))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)

def _apply_global_rot_frame(frame: Dict[str, Any], Rw: np.ndarray, mode: str) -> Dict[str, Any]:
    """
    Apply a global rotation to a pose.
    - mode="world": p' = Rw@p, R' = Rw@R
    - mode="basis": p' = Rw@p, R' = Rw@R@Rw.T
    """
    mode = str(mode).lower().strip()
    if mode not in ["world", "basis"]:
        mode = "world"
    Rw = np.asarray(Rw, dtype=np.float32).reshape(3, 3)
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        try:
            pos = np.asarray(v[0], dtype=np.float32).reshape(3)
            quat = np.asarray(v[1], dtype=np.float32).reshape(4)
            pos2 = (Rw @ pos).astype(np.float32)
            R = _quat_to_mat_wxyz(quat)
            R2 = (Rw @ R @ Rw.T) if mode == "basis" else (Rw @ R)
            quat2 = _mat_to_quat_wxyz(R2)
            out[k] = [pos2, quat2]
        except Exception:
            out[k] = v
    return out

def _make_debug_viz(only_raw: bool) -> Any:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise SystemExit(
            "❌ 缺少依赖 `matplotlib`，无法启用 debug 可视化。\n"
            "   解决：pip install matplotlib\n"
            f"   原始错误：{e}"
        )
    plt.ion()
    if bool(only_raw):
        fig = plt.figure(figsize=(6, 6))
        ax_raw = fig.add_subplot(111, projection="3d")
        ax_raw.set_title("raw")
        ax_raw.set_xlabel("X")
        ax_raw.set_ylabel("Y")
        ax_raw.set_zlabel("Z")
        return fig, [ax_raw]
    else:
        fig = plt.figure(figsize=(12, 4))
        ax_raw = fig.add_subplot(131, projection="3d")
        ax_xf = fig.add_subplot(132, projection="3d")
        ax_gmr = fig.add_subplot(133, projection="3d")
        axes = [ax_raw, ax_xf, ax_gmr]
        titles = ["raw", "xform", "gmr"]
        for ax, t in zip(axes, titles):
            ax.set_title(t)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        return fig, axes


def _viz_body_frame(ax: Any, frame: Dict[str, Any], scale: float, viewer_axes: bool) -> None:
    # If requested, match vmc_fk_viewer.py drawing exactly (same edges + axis mapping)
    if viewer_axes:
        try:
            from deploy_real import vmc_fk_viewer as _vmc_viewer  # type: ignore
            edges = list(getattr(_vmc_viewer, "DRAW_LINES", []))
        except Exception:
            edges = []
        ax.cla()
        ax.set_xlabel("X")
        ax.set_ylabel("Z (Depth)")
        ax.set_zlabel("Y (Up)")

        pts: Dict[str, np.ndarray] = {}
        for k, v in frame.items():
            try:
                pts[k] = np.asarray(v[0], dtype=np.float32).reshape(3) * float(scale)
            except Exception:
                continue
        if not pts or not edges:
            return
        for a, b in edges:
            pa = pts.get(a)
            pb = pts.get(b)
            if pa is None or pb is None:
                continue
            ax.plot([pa[0], pb[0]], [pa[2], pb[2]], [pa[1], pb[1]], c="r", marker="o")
        hips = pts.get("Hips", None)
        if hips is not None:
            r = 1.2 * float(scale)
            ax.set_xlim(hips[0] - r, hips[0] + r)
            ax.set_ylim(hips[2] - r, hips[2] + r)
            ax.set_zlim(hips[1] - r, hips[1] + r)
        return

    joint_aliases = {
        "Hips": ["Hips"],
        "Spine": ["Spine"],
        "Spine1": ["Spine1"],
        "Spine2": ["Spine2"],
        "Spine3": ["Spine3"],
        "Neck": ["Neck"],
        "Head": ["Head"],
        "LeftShoulder": ["LeftShoulder"],
        "LeftUpperArm": ["LeftUpperArm", "LeftArm"],
        "LeftLowerArm": ["LeftLowerArm", "LeftForeArm"],
        "LeftHand": ["LeftHand"],
        "RightShoulder": ["RightShoulder"],
        "RightUpperArm": ["RightUpperArm", "RightArm"],
        "RightLowerArm": ["RightLowerArm", "RightForeArm"],
        "RightHand": ["RightHand"],
        "LeftUpperLeg": ["LeftUpperLeg", "LeftUpLeg"],
        "LeftLowerLeg": ["LeftLowerLeg", "LeftLeg"],
        "LeftFoot": ["LeftFoot"],
        "LeftToe": ["LeftToe", "LeftToeBase"],
        "RightUpperLeg": ["RightUpperLeg", "RightUpLeg"],
        "RightLowerLeg": ["RightLowerLeg", "RightLeg"],
        "RightFoot": ["RightFoot"],
        "RightToe": ["RightToe", "RightToeBase"],
    }
    edges = [
        ("Hips", "Spine"),
        ("Spine", "Spine1"),
        ("Spine1", "Spine2"),
        ("Spine2", "Spine3"),
        ("Spine3", "Neck"),
        ("Neck", "Head"),
        ("Spine3", "LeftShoulder"),
        ("LeftShoulder", "LeftUpperArm"),
        ("LeftUpperArm", "LeftLowerArm"),
        ("LeftLowerArm", "LeftHand"),
        ("Spine3", "RightShoulder"),
        ("RightShoulder", "RightUpperArm"),
        ("RightUpperArm", "RightLowerArm"),
        ("RightLowerArm", "RightHand"),
        ("Hips", "LeftUpperLeg"),
        ("LeftUpperLeg", "LeftLowerLeg"),
        ("LeftLowerLeg", "LeftFoot"),
        ("LeftFoot", "LeftToe"),
        ("Hips", "RightUpperLeg"),
        ("RightUpperLeg", "RightLowerLeg"),
        ("RightLowerLeg", "RightFoot"),
        ("RightFoot", "RightToe"),
    ]

    def get_pos(name: str) -> Optional[np.ndarray]:
        for alias in joint_aliases.get(name, [name]):
            if alias in frame:
                return np.asarray(frame[alias][0], dtype=np.float32).reshape(3)
        return None

    ax.cla()
    if not viewer_axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    pts: Dict[str, np.ndarray] = {}
    for n in joint_aliases.keys():
        p = get_pos(n)
        if p is not None:
            pts[n] = p * float(scale)
    if not pts:
        return
    for a, b in edges:
        pa = pts.get(a)
        pb = pts.get(b)
        if pa is None or pb is None:
            continue
        if viewer_axes:
            ax.plot([pa[0], pb[0]], [pa[2], pb[2]], [pa[1], pb[1]], c="r", marker="o")
        else:
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], c="r", marker="o")
    hips = pts.get("Hips", None)
    if hips is not None:
        r = 1.2 * float(scale)
        ax.set_xlim(hips[0] - r, hips[0] + r)
        if viewer_axes:
            ax.set_ylim(hips[2] - r, hips[2] + r)
            ax.set_zlim(hips[1] - r, hips[1] + r)
        else:
            ax.set_ylim(hips[1] - r, hips[1] + r)
            ax.set_zlim(hips[2] - r, hips[2] + r)


def _debug_print_lr(frame: Dict[str, Any], label: str) -> None:
    pairs = [
        ("LeftShoulder", "RightShoulder"),
        ("LeftUpperArm", "RightUpperArm"),
        ("LeftLowerArm", "RightLowerArm"),
        ("LeftHand", "RightHand"),
        ("LeftUpperLeg", "RightUpperLeg"),
        ("LeftLowerLeg", "RightLowerLeg"),
        ("LeftFoot", "RightFoot"),
    ]
    def get_pos(name: str) -> Optional[np.ndarray]:
        if name in frame:
            return np.asarray(frame[name][0], dtype=np.float32).reshape(3)
        return None
    parts = []
    for l, r in pairs:
        pl = get_pos(l)
        pr = get_pos(r)
        if pl is None or pr is None:
            continue
        parts.append(f"{l}/{r}: Lx={pl[0]:+.3f} Rx={pr[0]:+.3f} Ly={pl[1]:+.3f} Ry={pr[1]:+.3f} Lz={pl[2]:+.3f} Rz={pr[2]:+.3f}")
    if parts:
        print(f"[LR:{label}] " + " | ".join(parts))


class VmcReceiver:
    def __init__(self, ip: str, port: int, invert_zw: bool) -> None:
        try:
            from pythonosc import dispatcher as osc_dispatcher  # type: ignore
            from pythonosc import osc_server  # type: ignore
        except Exception as e:
            raise SystemExit(
                "❌ 缺少依赖 `python-osc`，无法使用 VMC 接收。\n"
                "   解决：在你的运行环境里安装，例如：\n"
                "     - pip install python-osc\n"
                "     - 或 conda install -c conda-forge python-osc\n"
                f"   原始错误：{e}"
            )

        self._lock = threading.Lock()
        self._bones: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._bones_raw_xyzw: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._last_ts: float = 0.0
        self._seq: int = 0
        self._invert_zw = bool(invert_zw)

        disp = osc_dispatcher.Dispatcher()
        disp.map("/VMC/Ext/Bone/Pos", self._on_bone_pos)

        self._server = osc_server.ThreadingOSCUDPServer((str(ip), int(port)), disp)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _on_bone_pos(self, _addr: str, name: str, px: float, py: float, pz: float, qx: float, qy: float, qz: float, qw: float) -> None:
        ts = time.time()
        pos = np.array([float(px), float(py), float(pz)], dtype=np.float32)
        qx, qy, qz, qw = float(qx), float(qy), float(qz), float(qw)
        if self._invert_zw:
            qz = -qz
            qw = -qw
        quat = _vmc_quat_xyzw_to_wxyz(qx, qy, qz, qw, invert_zw=False)
        quat_raw = np.array([qx, qy, qz, qw], dtype=np.float32)
        key = _normalize_vmc_name(name)
        with self._lock:
            self._bones[key] = (pos, quat)
            self._bones_raw_xyzw[key] = (pos, quat_raw)
            self._last_ts = float(ts)
            self._seq += 1

    def snapshot(self, max_age_s: float) -> Tuple[Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]], int, float]:
        with self._lock:
            now = time.time()
            if self._last_ts <= 0 or (now - float(self._last_ts)) > float(max_age_s):
                return None, -1, 0.0
            return dict(self._bones), int(self._seq), float(self._last_ts)

    def snapshot_raw_xyzw(self, max_age_s: float) -> Tuple[Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]], int, float]:
        with self._lock:
            now = time.time()
            if self._last_ts <= 0 or (now - float(self._last_ts)) > float(max_age_s):
                return None, -1, 0.0
            return dict(self._bones_raw_xyzw), int(self._seq), float(self._last_ts)

    def close(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass


def _vmc_build_body_frame(
    bones: Dict[str, Tuple[np.ndarray, np.ndarray]],
    joint_names: list[str],
) -> Dict[str, Any]:
    def pick(cands: list[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        for c in cands:
            v = bones.get(c)
            if v is not None:
                return v
        return None

    def set_joint(out: Dict[str, Any], name: str, cands: list[str], fallback: Optional[str] = None) -> None:
        v = pick(cands)
        if v is None and fallback is not None:
            v = out.get(fallback)
        if v is None:
            return
        pos, quat = v
        out[str(name)] = [np.asarray(pos, dtype=np.float32).reshape(3), _safe_quat_wxyz(np.asarray(quat, dtype=np.float32).reshape(4))]

    # Candidate name sets (normalized)
    hips = ["hips", "hip", "root", "pelvis", "hiptracker"]
    spine = ["spine", "waist"]
    chest = ["chest"]
    upper_chest = ["upperchest"]
    neck = ["neck"]
    head = ["head"]

    r_shoulder = ["rightshoulder", "rightuppershoulder"]
    r_upper_arm = ["rightupperarm", "rightarm"]
    r_lower_arm = ["rightlowerarm", "rightforearm"]
    r_hand = ["righthand"]

    l_shoulder = ["leftshoulder", "leftuppershoulder"]
    l_upper_arm = ["leftupperarm", "leftarm"]
    l_lower_arm = ["leftlowerarm", "leftforearm"]
    l_hand = ["lefthand"]

    r_upper_leg = ["rightupperleg", "rightupleg", "rightthigh", "righthip"]
    r_lower_leg = ["rightlowerleg", "rightleg", "rightcalf"]
    r_foot = ["rightfoot", "rightankle"]
    r_toe = ["righttoe", "righttoes", "righttoeend"]

    l_upper_leg = ["leftupperleg", "leftupleg", "leftthigh", "lefthip"]
    l_lower_leg = ["leftlowerleg", "leftleg", "leftcalf"]
    l_foot = ["leftfoot", "leftankle"]
    l_toe = ["lefttoe", "lefttoes", "lefttoeend"]

    out: Dict[str, Any] = {}
    # Spine chain (fallbacks keep continuity)
    set_joint(out, "Hips", hips)
    set_joint(out, "Spine", spine + chest + hips, fallback="Hips")
    set_joint(out, "Spine1", chest + spine, fallback="Spine")
    set_joint(out, "Spine2", upper_chest + chest, fallback="Spine1")
    set_joint(out, "Spine3", upper_chest + chest + neck, fallback="Spine2")
    set_joint(out, "Neck", neck + upper_chest + chest, fallback="Spine3")
    set_joint(out, "Head", head + neck, fallback="Neck")

    set_joint(out, "RightShoulder", r_shoulder + upper_chest + chest, fallback="Spine3")
    set_joint(out, "RightUpperArm", r_upper_arm, fallback="RightShoulder")
    set_joint(out, "RightLowerArm", r_lower_arm, fallback="RightUpperArm")
    set_joint(out, "RightHand", r_hand, fallback="RightLowerArm")

    set_joint(out, "LeftShoulder", l_shoulder + upper_chest + chest, fallback="Spine3")
    set_joint(out, "LeftUpperArm", l_upper_arm, fallback="LeftShoulder")
    set_joint(out, "LeftLowerArm", l_lower_arm, fallback="LeftUpperArm")
    set_joint(out, "LeftHand", l_hand, fallback="LeftLowerArm")

    set_joint(out, "RightUpperLeg", r_upper_leg, fallback="Hips")
    set_joint(out, "RightLowerLeg", r_lower_leg, fallback="RightUpperLeg")
    set_joint(out, "RightFoot", r_foot, fallback="RightLowerLeg")
    set_joint(out, "RightToe", r_toe + r_foot, fallback="RightFoot")

    set_joint(out, "LeftUpperLeg", l_upper_leg, fallback="Hips")
    set_joint(out, "LeftLowerLeg", l_lower_leg, fallback="LeftUpperLeg")
    set_joint(out, "LeftFoot", l_foot, fallback="LeftLowerLeg")
    set_joint(out, "LeftToe", l_toe + l_foot, fallback="LeftFoot")

    # Ensure output order contains all expected joint_names (if provided)
    if joint_names:
        ordered: Dict[str, Any] = {}
        for n in joint_names:
            if n in out:
                ordered[n] = out[n]
        # Append any extra (rare)
        for k, v in out.items():
            if k not in ordered:
                ordered[k] = v
        return ordered
    return out

# ---------------------------------------------------------------------
# Safe idle (k: send_enabled=False) presets
# ---------------------------------------------------------------------
# When keyboard toggle is OFF (k), we publish a "safe idle" 35D mimic_obs.
# Add more presets here, then select via CLI: --safe_idle_pose_id 0/1/2...
#
# NOTE: Must be length 35.
SAFE_IDLE_BODY_35_PRESETS: dict[int, list[float]] = {
    # 0: original repo default (keeps backward compatibility)
    0: [
        -2.962986573041272e-06,
        6.836185035045111e-06,
        0.7900107971067252,
        0.026266981563484476,
        -0.07011304233181229,
        -0.00038564063739400495,
        0.21007653006396093,
        0.1255744557454361,
        0.5210019779740723,
        -0.087267,
        0.023696508296266388,
        -0.12259741578159437,
        0.18640974335249333,
        -0.1213838414703421,
        0.11017991599235927,
        -0.087267,
        -0.06074348170695354,
        0.10802879748679631,
        -0.14690420989255235,
        -0.06195140749854128,
        0.03492134295105836,
        -0.012934516116481467,
        0.012973065503571952,
        -0.09877424821663634,
        1.5735338678105346,
        -0.08846852951921763,
        -0.008568943127155513,
        -0.07037145190015832,
        -0.45191594425028536,
        -0.7548272891300677,
        0.07631181877180071,
        0.623873998918081,
        0.32440260037889024,
        -0.17081521970550126,
        0.2697219398563502,
    ],
    # 1: captured from Redis (action source) on your machine (2026-01-17)
    1: [
        -0.002425597382764927,
        0.0004014222794810171,
        0.789948578249186,
        -0.05286645234860116,
        -0.11395774381848182,
        -0.0020091780029543797,
        0.33550286925644013,
        0.07678254800339449,
        -0.11831599235723278,
        -0.087267,
        -0.1536621162766681,
        -0.039016535005063684,
        0.28263936593666483,
        -0.01999487086573224,
        -0.3918089438082317,
        -0.08726699999999998,
        -0.06775504509688593,
        0.0727761475591654,
        -0.09677870600760852,
        -0.0027568505266116657,
        0.07348304585982098,
        -0.10334908779279858,
        0.3160389030446376,
        0.07844298473038674,
        1.3008225711954524,
        0.6130673022421114,
        -0.2198179601159421,
        0.3438907117467236,
        -0.23448010297908417,
        -0.5483439694277361,
        -0.3146753829836872,
        0.910606700768848,
        -0.22716316478096404,
        -0.10501071874258898,
        -0.2864687400817216,
    ],
    2: [
        0.0, 0.0, 0.79, 0.004581602464116093, 0.054385222258041876, -0.01047197449952364, -0.1705406904220581, -0.011608824133872986, -0.08608310669660568, 0.2819371521472931, -0.13509835302829742, 0.028368590399622917, -0.15945219993591309, -0.011438383720815182, 0.09397093206644058, 0.2500985264778137, -0.12299267947673798, 0.033810943365097046, 0.01984678953886032, 0.04372693970799446, 0.04439987987279892, -0.052922338247299194, 0.3638530671596527, 0.018935075029730797, 1.2066316604614258, 0.0026964505668729544, -0.0038426220417022705, -0.05543806776404381, 0.016382435336709023, -0.3776109516620636, -0.07517704367637634, 1.2037315368652344, -0.03580886498093605, -0.07851681113243103, -0.011213400401175022
    ],
}


# 默认骨架（用于 SDK 初始化；直接复用 sync_capture.py 的值）
INITIAL_POSITION_BODY = [
        [0, 0, 1.022],
        [0.074, 0, 1.002],
        [0.097, 0, 0.593],
        [0.104, 0, 0.111],
        [0.114, 0.159, 0.005],
        [-0.074, 0, 1.002],
        [-0.097, 0.001, 0.593],
        [-0.104, 0, 0.111],
        [-0.114, 0.158, 0.004],
        [0, 0.033, 1.123],
        [0, 0.03, 1.246],
        [0, 0.014, 1.362],
        [0, -0.048, 1.475],
        [0, -0.048, 1.549],
        [0, -0.016, 1.682],
        [0.071, -0.061, 1.526],
        [0.178, -0.061, 1.526],
        [0.421, -0.061, 1.526],
        [0.682, -0.061, 1.526],
        [-0.071, -0.061, 1.526],
        [-0.178, -0.061, 1.526],
        [-0.421, -0.061, 1.526],
        [-0.682, -0.061, 1.526],
    ]

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


class KeyboardToggle:
    def __init__(
        self,
        enable: bool,
        toggle_send_key: str = "k",
        hold_key: str = "p",
        exit_key: str = "q",
        emergency_stop_key: str = "e",
        # gripper keys (Unitree action_hand_*): z/x for left, n/m for right
        left_close_key: str = "z",
        left_open_key: str = "x",
        right_close_key: str = "n",
        right_open_key: str = "m",
        hand_step: float = 0.05,
        backend: str = "stdin",  # stdin 或 evdev
        evdev_device: str = "auto",
        evdev_grab: bool = False,
    ) -> None:
        self.enable = bool(enable)
        self.toggle_send_key = (toggle_send_key or "k")[0]
        self.hold_key = (hold_key or "p")[0]
        self.exit_key = (exit_key or "q")[0]
        self.emergency_stop_key = (emergency_stop_key or "e")[0]
        self.left_close_key = (left_close_key or "z")[0]
        self.left_open_key = (left_open_key or "x")[0]
        self.right_close_key = (right_close_key or "n")[0]
        self.right_open_key = (right_open_key or "m")[0]
        self.hand_step = float(hand_step)
        self.backend = (backend or "stdin").strip().lower()
        self.evdev_device = str(evdev_device).strip() if evdev_device is not None else "auto"
        self.evdev_grab = bool(evdev_grab)

        self._send_enabled = True
        self._hold_enabled = False
        self._exit_requested = False
        # hand interpolation value in [0,1] (0=open, 1=closed)
        self._hand_left_position = 0.0
        self._hand_right_position = 0.0
        self._lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stdin_fd: Optional[int] = None
        self._stdin_old: Any = None
        self._evdev: Optional[EvdevHotkeys] = None

    def start(self) -> None:
        if not self.enable:
            return
        if self.backend in ["evdev", "both"]:
            if self._evdev is not None:
                return
            self._stop.clear()
            cfg = EvdevHotkeyConfig(device=self.evdev_device, grab=self.evdev_grab)
            self._evdev = EvdevHotkeys(cfg, callback=self._handle_key)
            print("[Keyboard] backend=evdev（全局热键，不依赖前台窗口/终端）")
            print(f"[Keyboard] evdev_device={cfg.device} grab={cfg.grab}")
            print(f"[Keyboard] 按 '{self.toggle_send_key}' 切换是否发送到 Redis（禁用时发送安全 idle）")
            print(f"[Keyboard] 按 '{self.hold_key}' 切换“保持当前位置”(冻结 action_body)")
            print(f"[Keyboard] 按 '{self.exit_key}' 退出程序")
            print(f"[Keyboard] 按 '{self.emergency_stop_key}' 紧急停止（kill sim2real）")
            print(
                f"[Keyboard] 夹爪手开合：左手 {self.left_close_key}/{self.left_open_key} (close/open)，右手 {self.right_close_key}/{self.right_open_key} (close/open)"
            )
            self._evdev.start()
            if self.backend == "evdev":
                return

        # default: stdin（需要终端在前台）
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except Exception:
            pass
        self._thread = None
        try:
            if self._evdev is not None:
                self._evdev.stop()
        except Exception:
            pass
        self._evdev = None

        try:
            if self._stdin_fd is not None and self._stdin_old is not None:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old)
        except Exception:
            pass

    def get_state(self) -> Tuple[bool, bool]:
        with self._lock:
            return bool(self._send_enabled), bool(self._hold_enabled)

    def get_extended_state(self) -> Tuple[bool, bool, bool, float, float]:
        with self._lock:
            return (
                bool(self._send_enabled),
                bool(self._hold_enabled),
                bool(self._exit_requested),
                float(self._hand_left_position),
                float(self._hand_right_position),
            )

    def request_exit(self) -> None:
        with self._lock:
            self._exit_requested = True

    def _emergency_stop(self) -> None:
        # mirror logic in xrobot_teleop_to_robot_w_hand_keyboard.py
        try:
            import subprocess

            print("[EMERGENCY STOP] Killing sim2real.sh process...")
            subprocess.run(["pkill", "-f", "sim2real.sh"], capture_output=True, text=True, timeout=5)
            subprocess.run(
                ["pkill", "-f", "server_low_level_g1_real_future.py"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as e:
            print(f"[EMERGENCY STOP] Error executing pkill: {e}")

    def _handle_key(self, ch: str) -> None:
        """统一处理按键（stdin/evdev 复用）。"""
        ch = (ch or "")[0]
        if ch == self.toggle_send_key:
            with self._lock:
                self._send_enabled = not self._send_enabled
                if not self._send_enabled:
                    self._hold_enabled = False
                enabled = self._send_enabled
            print(f"[Keyboard] send_enabled => {enabled}")
        elif ch == self.hold_key:
            with self._lock:
                if not self._send_enabled:
                    self._hold_enabled = False
                else:
                    self._hold_enabled = not self._hold_enabled
                enabled = self._hold_enabled
            print(f"[Keyboard] hold_position_enabled => {enabled}")
        elif ch == self.exit_key:
            with self._lock:
                self._exit_requested = True
            print("[Keyboard] exit requested")
        elif ch == self.emergency_stop_key:
            self._emergency_stop()
        elif ch == self.left_close_key:
            with self._lock:
                self._hand_left_position = float(min(1.0, self._hand_left_position + self.hand_step))
                v = self._hand_left_position
            print(f"[Keyboard] left hand closing => {v:.2f}")
        elif ch == self.left_open_key:
            with self._lock:
                self._hand_left_position = float(max(0.0, self._hand_left_position - self.hand_step))
                v = self._hand_left_position
            print(f"[Keyboard] left hand opening => {v:.2f}")
        elif ch == self.right_close_key:
            with self._lock:
                self._hand_right_position = float(min(1.0, self._hand_right_position + self.hand_step))
                v = self._hand_right_position
            print(f"[Keyboard] right hand closing => {v:.2f}")
        elif ch == self.right_open_key:
            with self._lock:
                self._hand_right_position = float(max(0.0, self._hand_right_position - self.hand_step))
                v = self._hand_right_position
            print(f"[Keyboard] right hand opening => {v:.2f}")

    def _loop(self) -> None:
        try:
            fd = sys.stdin.fileno()
            self._stdin_fd = fd
            self._stdin_old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            print(f"[Keyboard] 按 '{self.toggle_send_key}' 切换是否发送到 Redis（禁用时发送安全 idle）")
            print(f"[Keyboard] 按 '{self.hold_key}' 切换“保持当前位置”(冻结 action_body)")
            print(f"[Keyboard] 按 '{self.exit_key}' 退出程序")
            print(f"[Keyboard] 按 '{self.emergency_stop_key}' 紧急停止（kill sim2real）")
            print(
                f"[Keyboard] 夹爪手开合：左手 {self.left_close_key}/{self.left_open_key} (close/open)，右手 {self.right_close_key}/{self.right_open_key} (close/open)"
            )

            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                self._handle_key(ch)
        except Exception as e:
            print(f"[Keyboard] 键盘监听不可用：{e}")


class SmoothFilter:
    def __init__(self, enable: bool, window_size: int = 5) -> None:
        self.enable = bool(enable)
        self.window_size = max(1, int(window_size))
        self._buf: list[np.ndarray] = []

    def reset(self) -> None:
        self._buf = []

    def apply(self, x: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if x is None:
            return None
        if not self.enable:
            return x
        v = np.asarray(x, dtype=float).copy()
        self._buf.append(v)
        if len(self._buf) > self.window_size:
            self._buf.pop(0)
        if len(self._buf) >= 2:
            return np.mean(np.stack(self._buf, axis=0), axis=0)
        return v


def _hand_pose_from_value(robot_name: str, left_val: float, right_val: float, pinch_mode: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce 7D action_hand pose for left/right using DEFAULT_HAND_POSE.
    left_val/right_val in [0,1], where 0=open, 1=closed.
    """
    from data_utils.params import DEFAULT_HAND_POSE

    lv = float(np.clip(left_val, 0.0, 1.0))
    rv = float(np.clip(right_val, 0.0, 1.0))
    cfg = DEFAULT_HAND_POSE[robot_name]
    if not pinch_mode:
        left_open = cfg["left"]["open"]
        left_closed = cfg["left"]["close"]
        right_open = cfg["right"]["open"]
        right_closed = cfg["right"]["close"]
    else:
        # follow xrobot_teleop_to_robot_w_hand_keyboard.py logic
        left_fully_open = cfg["left"]["open_pinch"]
        left_fully_closed = cfg["left"]["close_pinch"]
        right_fully_open = cfg["right"]["open_pinch"]
        right_fully_closed = cfg["right"]["close_pinch"]
        ratio_open = 0.8
        ratio_closed = 0.0
        left_open = left_fully_open * ratio_open + (1 - ratio_open) * left_fully_closed
        left_closed = left_fully_open * ratio_closed + (1 - ratio_closed) * left_fully_closed
        right_open = right_fully_open * ratio_open + (1 - ratio_open) * right_fully_closed
        right_closed = right_fully_open * ratio_closed + (1 - ratio_closed) * right_fully_closed

    left_pose = left_open + (left_closed - left_open) * lv
    right_pose = right_open + (right_closed - right_open) * rv
    return np.asarray(left_pose, dtype=float), np.asarray(right_pose, dtype=float)


def _ease(alpha: float, ease: str = "cosine") -> float:
    """Ramp easing curve. alpha in [0,1]."""
    a = float(np.clip(alpha, 0.0, 1.0))
    if str(ease).lower() == "linear":
        return a
    # cosine ease-in-out
    return 0.5 - 0.5 * float(np.cos(np.pi * a))


def extract_mimic_obs_whole_body(qpos: np.ndarray, last_qpos: np.ndarray, dt: float) -> np.ndarray:
    from data_utils.rot_utils import euler_from_quaternion_np, quat_diff_np, quat_rotate_inverse_np

    root_pos, last_root_pos = qpos[0:3], last_qpos[0:3]
    root_quat, last_root_quat = qpos[3:7], last_qpos[3:7]
    robot_joints = qpos[7:].copy()

    base_vel = (root_pos - last_root_pos) / dt
    base_ang_vel = quat_diff_np(last_root_quat, root_quat, scalar_first=True) / dt
    roll, pitch, yaw = euler_from_quaternion_np(root_quat.reshape(1, -1), scalar_first=True)
    base_vel_local = quat_rotate_inverse_np(root_quat, base_vel, scalar_first=True)
    base_ang_vel_local = quat_rotate_inverse_np(root_quat, base_ang_vel, scalar_first=True)

    mimic_obs = np.concatenate(
        [
            base_vel_local[:2],
            root_pos[2:3],
            roll,
            pitch,
            base_ang_vel_local[2:3],
            robot_joints,
        ]
    )
    return mimic_obs


def _import_xdmocap_sdk():
    sdk_dir = _repo_root() / "xdmocap" / "DataRead_Python_Demo"
    if not sdk_dir.exists():
        raise FileNotFoundError(f"找不到 XDMocap SDK 路径: {sdk_dir}")
    if str(sdk_dir) not in sys.path:
        sys.path.insert(0, str(sdk_dir))
    from vdmocapsdk_dataread import (  # type: ignore
        MocapData,
        udp_close,
        udp_is_open,
        udp_open,
        udp_recv_mocap_data,
        udp_remove,
        udp_send_request_connect,
        udp_set_position_in_initial_tpose,
    )
    from vdmocapsdk_nodelist import (  # type: ignore
        LENGTH_BODY,
        LENGTH_HAND,
        NAMES_JOINT_BODY,
        NAMES_JOINT_HAND_LEFT,
        NAMES_JOINT_HAND_RIGHT,
    )

    return {
        "MocapData": MocapData,
        "udp_open": udp_open,
        "udp_is_open": udp_is_open,
        "udp_close": udp_close,
        "udp_remove": udp_remove,
        "udp_send_request_connect": udp_send_request_connect,
        "udp_set_position_in_initial_tpose": udp_set_position_in_initial_tpose,
        "udp_recv_mocap_data": udp_recv_mocap_data,
        "LENGTH_BODY": int(LENGTH_BODY),
        "LENGTH_HAND": int(LENGTH_HAND),
        "NAMES_JOINT_BODY": list(NAMES_JOINT_BODY),
        "NAMES_JOINT_HAND_LEFT": list(NAMES_JOINT_HAND_LEFT),
        "NAMES_JOINT_HAND_RIGHT": list(NAMES_JOINT_HAND_RIGHT),
    }


def _build_body_frame_from_sdk(mocap_data: Any, joint_names: list[str], length_body: int) -> Dict[str, Any]:
    fr: Dict[str, Any] = {}
    for i in range(length_body):
        name = joint_names[i]
        p = np.array(
            [
                float(mocap_data.position_body[i][0]),
                float(mocap_data.position_body[i][1]),
                float(mocap_data.position_body[i][2]),
            ],
            dtype=np.float32,
        )
        q = np.array(
            [
                float(mocap_data.quaternion_body[i][0]),
                float(mocap_data.quaternion_body[i][1]),
                float(mocap_data.quaternion_body[i][2]),
                float(mocap_data.quaternion_body[i][3]),
            ],
            dtype=np.float32,
        )
        fr[name] = [p, _safe_quat_wxyz(q)]
    return fr


def _build_hand_frame_from_sdk(
    mocap_data: Any,
    joint_names: list[str],
    *,
    is_left: bool,
    length_hand: int,
) -> Dict[str, Any]:
    """
    从 SDK hand arrays 构造 frame。SDK hand joint name 已是 BVH 风格：
      LeftHand, LeftThumbFinger, LeftIndexFinger1, ... (20 joints)
    """
    fr: Dict[str, Any] = {}
    pos_arr = mocap_data.position_lHand if is_left else mocap_data.position_rHand
    quat_arr = mocap_data.quaternion_lHand if is_left else mocap_data.quaternion_rHand
    for i in range(length_hand):
        name = str(joint_names[i])
        p = np.array([float(pos_arr[i][0]), float(pos_arr[i][1]), float(pos_arr[i][2])], dtype=np.float32)
        q = np.array([float(quat_arr[i][0]), float(quat_arr[i][1]), float(quat_arr[i][2]), float(quat_arr[i][3])], dtype=np.float32)
        fr[name] = [p, _safe_quat_wxyz(q)]
    return fr


def _make_tracking_joint(pos: np.ndarray) -> list:
    # [[x,y,z], [qw,qx,qy,qz]]，Wuji server 主要用 position；quat 用单位四元数占位即可
    p = np.asarray(pos, dtype=np.float32).reshape(3)
    return [p.reshape(-1).tolist(), [1.0, 0.0, 0.0, 0.0]]


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Quaternion (wxyz) -> 3x3 rotation matrix.
    Mirror MATLAB `T_SY.m` exactly.
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


def _hand_joint_order_names(prefix: str) -> list[str]:
    """
    Official vendor 20-joint hand order (matches MATLAB Quat2Position*Hand.m):
      1 Hand
      2-4 ThumbFinger / ThumbFinger1 / ThumbFinger2
      5-8 IndexFinger..IndexFinger3
      9-12 MiddleFinger..MiddleFinger3
      13-16 RingFinger..RingFinger3
      17-20 PinkyFinger..PinkyFinger3
    """
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
    end_site_scale: Union[float, np.ndarray] = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Port of vendor MATLAB:
      - Quat2PositionLHand.m / Quat2PositionRHand.m
      - T_SY.m

    Inputs:
      - quats_wxyz: (20,4) global quats in the same world coordinate system as root_pos/bone_init_pos
      - root_pos: (3,)
      - bone_init_pos: (20,3) initial T-pose joint positions (same order)
    Outputs:
      - pos: (20,3) joint positions
      - pos_end: (5,3) fingertip end-sites (thumb/index/middle/ring/little)
    """
    q = np.asarray(quats_wxyz, dtype=np.float32).reshape(20, 4)
    root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3)
    bone = np.asarray(bone_init_pos, dtype=np.float32).reshape(20, 3)

    # Parent index list (0-based), matches MATLAB FN = [...]+1
    parent = np.array([0, 0, 1, 2, 0, 4, 5, 6, 0, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18], dtype=np.int32)

    bone_ver = bone - bone[parent]
    pos = np.zeros((20, 3), dtype=np.float32)
    pos[0] = root_pos
    for j in range(1, 20):
        pidx = int(parent[j])
        R = _quat_wxyz_to_rotmat(q[pidx])
        pos[j] = pos[pidx] + (R @ bone_ver[j].reshape(3, 1)).reshape(3)

    # End sites: use last segment vector * end_site_scale, rotated by the distal joint quat
    end_joint = np.array([3, 7, 11, 15, 19], dtype=np.int32)  # ThumbFinger2, IndexFinger3, ...
    prev_joint = np.array([2, 6, 10, 14, 18], dtype=np.int32)
    # end_site_scale can be:
    # - scalar: apply to all 5 fingertips
    # - (5,) array: per-finger scale in order [thumb,index,middle,ring,little]
    if isinstance(end_site_scale, (list, tuple, np.ndarray)):
        s = np.asarray(end_site_scale, dtype=np.float32).reshape(-1)
        if s.size != 5:
            raise ValueError(f"end_site_scale must be scalar or 5-vector, got shape={s.shape}")
        bone_ver_end = (bone[end_joint] - bone[prev_joint]) * s.reshape(5, 1)
    else:
        bone_ver_end = (bone[end_joint] - bone[prev_joint]) * float(end_site_scale)
    pos_end = np.zeros((5, 3), dtype=np.float32)
    for k in range(5):
        j = int(end_joint[k])
        R = _quat_wxyz_to_rotmat(q[j])
        pos_end[k] = pos[j] + (R @ bone_ver_end[k].reshape(3, 1)).reshape(3)
    return pos, pos_end


def _parse_hand_fk_end_site_scale(v: str) -> Union[float, np.ndarray]:
    """
    Parse --hand_fk_end_site_scale:
      - "0.8" -> 0.8
      - "0.858,0.882,0.882,0.882,0.882" -> np.ndarray shape (5,)
    """
    s = str(v).strip()
    if "," not in s:
        return float(s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("hand_fk_end_site_scale expects 1 value or 5 comma-separated values")
    try:
        arr = np.asarray([float(x) for x in parts], dtype=np.float32)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid hand_fk_end_site_scale: {e}") from e
    return arr


def _hand_to_tracking26(
    frame: Dict[str, Any],
    side: str,
    *,
    pos_override: Optional[Dict[str, np.ndarray]] = None,
    tip_override: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    把 BVH 风格手部骨骼（LeftHand/LeftIndexFinger...）映射到 Wuji 输入 26D dict key。
    对齐：deploy_real/replay_3d_bvh_wuji_to_redis.py 的 _bvh_hand_to_tracking26（此处不使用 EndSite）。
    """
    side = side.lower()
    assert side in ["left", "right"]
    pfx = "Left" if side == "left" else "Right"
    out: Dict[str, Any] = {}

    def get_pos(name: str) -> np.ndarray:
        if pos_override is not None and name in pos_override:
            return np.asarray(pos_override[name], dtype=np.float32).reshape(3)
        v = frame.get(name, None)
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            return np.asarray(v[0], dtype=np.float32).reshape(3)
        return np.zeros(3, dtype=np.float32)

    def pick(*names: str) -> str:
        for n in names:
            if n in frame:
                return n
        return names[0]

    # Wuji 手指长度缩放（thumb/index/middle/ring/little）
    # 这里按“骨节段方向”缩放，而不是直接对某个 tip 点做经验性缩放：
    # - finger  -> finger1
    # - finger1 -> finger2
    # - finger2 -> finger3
    # thumb 只有 finger -> finger1 -> finger2 两段
    # WUJI_FINGER_TIP_SCALING = [1.176453, 1.1299, 1.06, 1.1145, 1.243]  # xdmocap_ours
    WUJI_FINGER_TIP_SCALING = [1.0, 1.0, 1.0, 1.0, 1.0]  # xdmocap_ours

    def scale_chain_4seg(o: np.ndarray, m: np.ndarray, p: np.ndarray, i: np.ndarray, d: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        给 5 个点 (o,m,p,i,d) 做 4 段缩放重建：
          m' = o + s*(m-o)
          p' = m' + s*(p-m)
          i' = p' + s*(i-p)
          d' = i' + s*(d-i)
        """
        m2 = o + s * (m - o)
        p2 = m2 + s * (p - m)
        i2 = p2 + s * (i - p)
        d2 = i2 + s * (d - i)
        return m2, p2, i2, d2

    def scale_chain_3seg(o: np.ndarray, m: np.ndarray, p: np.ndarray, d: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        thumb 用：给 4 个点 (o,m,p,d) 做 3 段缩放重建：
          m' = o + s*(m-o)
          p' = m' + s*(p-m)
          d' = p' + s*(d-p)
        """
        m2 = o + s * (m - o)
        p2 = m2 + s * (p - m)
        d2 = p2 + s * (d - p)
        return m2, p2, d2

    # wrist/palm：没有 Palm，用 Hand 代替
    hand_name = pick(f"{pfx}Hand")
    hand_pos = get_pos(hand_name)
    out[f"{pfx}HandWrist"] = _make_tracking_joint(hand_pos)
    out[f"{pfx}HandPalm"] = _make_tracking_joint(hand_pos)

    # thumb (3 joints)
    th_o = hand_pos
    th_m = get_pos(pick(f"{pfx}ThumbFinger"))
    th_p = get_pos(pick(f"{pfx}ThumbFinger1", f"{pfx}ThumbFinger"))
    th_d = get_pos(pick(f"{pfx}ThumbFinger2", f"{pfx}ThumbFinger1"))
    th_s = float(WUJI_FINGER_TIP_SCALING[0])
    th_m2, th_p2, th_d2 = scale_chain_3seg(th_o, th_m, th_p, th_d, th_s)
    out[f"{pfx}HandThumbMetacarpal"] = _make_tracking_joint(th_m2)
    out[f"{pfx}HandThumbProximal"] = _make_tracking_joint(th_p2)
    out[f"{pfx}HandThumbDistal"] = _make_tracking_joint(th_d2)
    if tip_override is not None and f"{pfx}HandThumbTip" in tip_override:
        out[f"{pfx}HandThumbTip"] = _make_tracking_joint(tip_override[f"{pfx}HandThumbTip"])
    else:
        out[f"{pfx}HandThumbTip"] = _make_tracking_joint(th_d2)

    # index
    in_o = hand_pos
    in_m = get_pos(pick(f"{pfx}IndexFinger"))
    in_p = get_pos(pick(f"{pfx}IndexFinger1", f"{pfx}IndexFinger"))
    in_i = get_pos(pick(f"{pfx}IndexFinger2", f"{pfx}IndexFinger1"))
    in_d = get_pos(pick(f"{pfx}IndexFinger3", f"{pfx}IndexFinger2"))
    in_s = float(WUJI_FINGER_TIP_SCALING[1])
    in_m2, in_p2, in_i2, in_d2 = scale_chain_4seg(in_o, in_m, in_p, in_i, in_d, in_s)
    out[f"{pfx}HandIndexMetacarpal"] = _make_tracking_joint(in_m2)
    out[f"{pfx}HandIndexProximal"] = _make_tracking_joint(in_p2)
    out[f"{pfx}HandIndexIntermediate"] = _make_tracking_joint(in_i2)
    out[f"{pfx}HandIndexDistal"] = _make_tracking_joint(in_d2)
    if tip_override is not None and f"{pfx}HandIndexTip" in tip_override:
        out[f"{pfx}HandIndexTip"] = _make_tracking_joint(tip_override[f"{pfx}HandIndexTip"])
    else:
        out[f"{pfx}HandIndexTip"] = _make_tracking_joint(in_d2)

    # middle
    mi_o = hand_pos
    mi_m = get_pos(pick(f"{pfx}MiddleFinger"))
    mi_p = get_pos(pick(f"{pfx}MiddleFinger1", f"{pfx}MiddleFinger"))
    mi_i = get_pos(pick(f"{pfx}MiddleFinger2", f"{pfx}MiddleFinger1"))
    mi_d = get_pos(pick(f"{pfx}MiddleFinger3", f"{pfx}MiddleFinger2"))
    mi_s = float(WUJI_FINGER_TIP_SCALING[2])
    mi_m2, mi_p2, mi_i2, mi_d2 = scale_chain_4seg(mi_o, mi_m, mi_p, mi_i, mi_d, mi_s)
    out[f"{pfx}HandMiddleMetacarpal"] = _make_tracking_joint(mi_m2)
    out[f"{pfx}HandMiddleProximal"] = _make_tracking_joint(mi_p2)
    out[f"{pfx}HandMiddleIntermediate"] = _make_tracking_joint(mi_i2)
    out[f"{pfx}HandMiddleDistal"] = _make_tracking_joint(mi_d2)
    if tip_override is not None and f"{pfx}HandMiddleTip" in tip_override:
        out[f"{pfx}HandMiddleTip"] = _make_tracking_joint(tip_override[f"{pfx}HandMiddleTip"])
    else:
        out[f"{pfx}HandMiddleTip"] = _make_tracking_joint(mi_d2)

    # ring
    ri_o = hand_pos
    ri_m = get_pos(pick(f"{pfx}RingFinger"))
    ri_p = get_pos(pick(f"{pfx}RingFinger1", f"{pfx}RingFinger"))
    ri_i = get_pos(pick(f"{pfx}RingFinger2", f"{pfx}RingFinger1"))
    ri_d = get_pos(pick(f"{pfx}RingFinger3", f"{pfx}RingFinger2"))
    ri_s = float(WUJI_FINGER_TIP_SCALING[3])
    ri_m2, ri_p2, ri_i2, ri_d2 = scale_chain_4seg(ri_o, ri_m, ri_p, ri_i, ri_d, ri_s)
    out[f"{pfx}HandRingMetacarpal"] = _make_tracking_joint(ri_m2)
    out[f"{pfx}HandRingProximal"] = _make_tracking_joint(ri_p2)
    out[f"{pfx}HandRingIntermediate"] = _make_tracking_joint(ri_i2)
    out[f"{pfx}HandRingDistal"] = _make_tracking_joint(ri_d2)
    if tip_override is not None and f"{pfx}HandRingTip" in tip_override:
        out[f"{pfx}HandRingTip"] = _make_tracking_joint(tip_override[f"{pfx}HandRingTip"])
    else:
        out[f"{pfx}HandRingTip"] = _make_tracking_joint(ri_d2)

    # little/pinky
    li_o = hand_pos
    li_m = get_pos(pick(f"{pfx}PinkyFinger"))
    li_p = get_pos(pick(f"{pfx}PinkyFinger1", f"{pfx}PinkyFinger"))
    li_i = get_pos(pick(f"{pfx}PinkyFinger2", f"{pfx}PinkyFinger1"))
    li_d = get_pos(pick(f"{pfx}PinkyFinger3", f"{pfx}PinkyFinger2"))
    li_s = float(WUJI_FINGER_TIP_SCALING[4])
    li_m2, li_p2, li_i2, li_d2 = scale_chain_4seg(li_o, li_m, li_p, li_i, li_d, li_s)
    out[f"{pfx}HandLittleMetacarpal"] = _make_tracking_joint(li_m2)
    out[f"{pfx}HandLittleProximal"] = _make_tracking_joint(li_p2)
    out[f"{pfx}HandLittleIntermediate"] = _make_tracking_joint(li_i2)
    out[f"{pfx}HandLittleDistal"] = _make_tracking_joint(li_d2)
    if tip_override is not None and f"{pfx}HandLittleTip" in tip_override:
        out[f"{pfx}HandLittleTip"] = _make_tracking_joint(tip_override[f"{pfx}HandLittleTip"])
    else:
        out[f"{pfx}HandLittleTip"] = _make_tracking_joint(li_d2)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XDMocap real-time teleop (BODY ONLY) -> Redis")
    p.add_argument(
        "--body_source",
        choices=["xdmocap", "vmc"],
        default="xdmocap",
        help="Body source: xdmocap (SDK UDP) or vmc (OSC, e.g. SlimeVR).",
    )
    p.add_argument("--dst_ip", type=str, default="192.168.1.112", help="XDMocap data sender IP")
    p.add_argument("--dst_port", type=int, default=7000, help="XDMocap data sender port")
    p.add_argument("--local_port", type=int, default=0, help="Local UDP port (0 for auto)")
    p.add_argument("--mocap_index", type=int, default=0, help="SDK mocap index")
    p.add_argument("--world_space", type=int, default=0, help="SDK world space (0=WS_Geo)")

    # VMC (OSC) body input
    p.add_argument("--vmc_ip", type=str, default="0.0.0.0", help="VMC OSC listen IP")
    p.add_argument("--vmc_port", type=int, default=39539, help="VMC OSC listen port (default 39539)")
    p.add_argument(
        "--vmc_timeout_s",
        type=float,
        default=0.5,
        help="If no VMC packet within this time, treat body as inactive (seconds).",
    )
    p.add_argument("--vmc_rot_mode", choices=["local", "global"], default="local", help="VMC rotations are local/global")
    p.add_argument("--vmc_invert_zw", action="store_true", help="Invert VMC quaternion z/w (Unity coords)")
    p.add_argument("--vmc_no_invert_zw", action="store_true", help="Do not invert VMC quaternion z/w")
    p.add_argument("--vmc_use_fk", action="store_true", help="Reconstruct body positions via FK")
    p.add_argument("--vmc_bvh_path", type=str, default="bvh-recording.bvh", help="BVH for FK offsets")
    p.add_argument("--vmc_bvh_scale", type=float, default=1.0, help="BVH offset scale for FK")
    p.add_argument("--vmc_fk_skeleton", choices=["bvh", "std"], default="bvh", help="FK skeleton source")
    p.add_argument("--vmc_bvh_lengths_only", action="store_true", help="Use BVH bone lengths but keep STD directions")
    p.add_argument("--vmc_swap_lr", action="store_true", help="Swap left/right VMC bones")
    p.add_argument("--vmc_bvh_axis_swap", type=str, default="xyz", help="Axis swap for BVH offsets (e.g. xzy, yxz)")
    p.add_argument("--vmc_bvh_axis_flip", type=str, default="", help="Axis flips for BVH offsets (e.g. x, yz)")
    p.add_argument("--vmc_viewer_bone_axis_override", type=str, default="", help="Pass to vmc_fk_viewer --bone_axis_override")
    p.add_argument("--vmc_bvh_axis_auto", action="store_true", help="Auto cycle BVH offset axis swap/flip")
    p.add_argument("--vmc_bvh_axis_cycle_s", type=float, default=2.0, help="Seconds per axis combo when auto cycling")
    p.add_argument("--vmc_bvh_axis_cycle_once", action="store_true", help="Cycle axis combos once then stop")
    p.add_argument("--vmc_use_viewer_fk", action="store_true", help="Use deploy_real/vmc_fk_viewer.py FK directly")

    # Optional: split sources (body and hand can come from different senders / mocap_index).
    # If not provided, they fall back to --dst_ip/--dst_port/--mocap_index.
    p.add_argument("--dst_ip_body", type=str, default="", help="Body data sender IP (override --dst_ip)")
    p.add_argument("--dst_port_body", type=int, default=0, help="Body data sender port (override --dst_port)")
    p.add_argument("--mocap_index_body", type=int, default=-1, help="SDK mocap index for body (override --mocap_index)")
    p.add_argument("--dst_ip_hand", type=str, default="", help="Hand data sender IP (override --dst_ip)")
    p.add_argument("--dst_port_hand", type=int, default=0, help="Hand data sender port (override --dst_port)")
    p.add_argument("--mocap_index_hand", type=int, default=-1, help="SDK mocap index for hand (override --mocap_index)")
    p.add_argument(
        "--hand_source_timeout_s",
        type=float,
        default=0.5,
        help="If hand source hasn't updated for this many seconds, publish hand_tracking_* as inactive (when hands!=none).",
    )

    p.add_argument("--redis_ip", type=str, default="localhost", help="Redis host")
    p.add_argument("--format", choices=["lafan1", "nokov"], default="nokov", help="BVH skeleton naming convention for GMR")
    p.add_argument("--actual_human_height", type=float, default=1.45, help="Human height for GMR scaling")
    p.add_argument("--offset_to_ground", action="store_true", help="Offset to ground before retarget")

    # keep naming consistent with replay scripts
    p.add_argument("--csv_geo_to_bvh_official", action="store_true", help="Apply vendor official Geo->BVH mapping (pos+quat)")
    p.add_argument("--csv_apply_bvh_rotation", action="store_true", help="Apply BVH->GMR fixed axis rotation (BVH raw world -> GMR world)")
    p.add_argument("--csv_bvh_rot_mode", type=str, default="global", help="How to apply BVH rotation: global|basis")
    p.add_argument("--csv_bvh_rot_tweak", type=str, default="none", help="Extra tweak on top of --csv_apply_bvh_rotation: none|rx180|ry180|rz180")
    p.add_argument("--csv_bvh_rot_tweak_order", type=str, default="post", help="Tweak order for csv_bvh_rot_tweak: pre|post")
    p.add_argument("--csv_bvh_apply_pos", action="store_true", help="Apply BVH rotation to positions (default on)")
    p.add_argument("--csv_bvh_apply_quat", action="store_true", help="Apply BVH rotation to quaternions (default on)")
    p.add_argument("--hands", choices=["left", "right", "both", "none"], default="both", help="Publish hand_tracking_* and control Wuji hands")
    p.add_argument(
        "--hand_no_csv_transform",
        action="store_true",
        help="Do NOT apply --csv_geo_to_bvh_official and --csv_apply_bvh_rotation to HAND joints (LeftHand/RightHand...). "
             "Body joints are still transformed for GMR retarget. Useful when hand positions look wrong due to axis mapping.",
    )
    p.add_argument(
        "--publish_bvh_hand",
        action="store_true",
        help="Also publish BVH-style hand joint dict (LeftHand/LeftIndexFinger...) to Redis for debugging/visualization",
    )

    # Optional: use vendor FK to compute hand joint positions + fingertip EndSites (more accurate tips)
    p.add_argument("--hand_fk", action="store_true", help="Compute hand joint positions via vendor FK (uses quats + T-pose offsets); tips use EndSite")
    p.add_argument(
        "--hand_fk_end_site_scale",
        type=_parse_hand_fk_end_site_scale,
        default="0.8",
        help="EndSite length scale. Accepts scalar (e.g. 0.8) or 5 values for [thumb,index,middle,ring,little], "
             "e.g. 0.858,0.882,0.882,0.882,0.882 (vendor MATLAB default=0.8).",
    )

    # Optional controls (port from xrobot teleop; default OFF to keep behavior stable)
    p.add_argument("--control_neck", action="store_true", help="Publish action_neck_* (yaw/pitch) computed from human_head_to_robot_neck")
    p.add_argument("--neck_retarget_scale", type=float, default=1.5, help="Scale factor for neck yaw/pitch")
    p.add_argument("--control_gripper_hand_action", action="store_true", help="Publish action_hand_left/right_* (7D) using keyboard open/close")
    p.add_argument("--pinch_mode", action="store_true", help="Use pinch-mode hand pose mapping (like xrobot teleop)")
    p.add_argument("--hand_step", type=float, default=0.05, help="Per key press step for gripper open/close in [0,1]")
    p.add_argument("--smooth", action="store_true", help="Enable sliding-window smoothing for 35D mimic_obs")
    p.add_argument("--smooth_window_size", type=int, default=5, help="Smoothing window size (frames)")
    p.add_argument("--viz", action="store_true", help="Enable MuJoCo visualization (viewer)")
    p.add_argument("--record_video", action="store_true", help="Record MuJoCo viewer to mp4 (requires --viz)")
    p.add_argument("--measure_fps", type=int, default=0, help="Print FPS stats (0=off, 1=on, like teleop.sh)")

    # ---------------------------------------------------------------------
    # Wuji hand sim visualization (spawn processes; implemented here, NOT in bash)
    # ---------------------------------------------------------------------
    p.add_argument(
        "--wuji_hand_sim_viz",
        action="store_true",
        help="启动 Wuji 手 MuJoCo sim 可视化（从 Redis 读 hand_tracking_*，做后处理并驱动手模型）。",
    )
    p.add_argument(
        "--wuji_hand_sim_sides",
        choices=["left", "right", "both", "none"],
        default="both",
        help="启动哪只手的 sim（默认 both；none 等同不启动）。",
    )
    p.add_argument(
        "--wuji_hand_sim_target_fps",
        type=float,
        default=60.0,
        help="Wuji 手 sim 更新频率（Hz，默认 60）。",
    )
    # sim viz: choose DexPilot(retarget) vs GeoRT(model inference)
    # 参数名对齐 wuji_hand_model_deploy.sh / deploy2.py
    p.add_argument("--wuji_hand_sim_use_model", action="store_true", help="Wuji 手 sim 端使用 GeoRT 模型推理（默认不启用：DexPilot retarget）")
    p.add_argument("--wuji_hand_sim_policy_tag", type=str, default="geort_filter_wuji", help="sim 端 GeoRT 模型 tag（--wuji_hand_sim_use_model）")
    p.add_argument("--wuji_hand_sim_policy_epoch", type=int, default=-1, help="sim 端 GeoRT 模型 epoch（--wuji_hand_sim_use_model）")
    p.add_argument("--wuji_hand_sim_policy_tag_left", type=str, default="", help="左手 tag（可选；空则用 --wuji_hand_sim_policy_tag）")
    p.add_argument("--wuji_hand_sim_policy_epoch_left", type=int, default=-999999, help="左手 epoch（可选；-999999 表示用 --wuji_hand_sim_policy_epoch）")
    p.add_argument("--wuji_hand_sim_policy_tag_right", type=str, default="", help="右手 tag（可选；空则用 --wuji_hand_sim_policy_tag）")
    p.add_argument("--wuji_hand_sim_policy_epoch_right", type=int, default=-999999, help="右手 epoch（可选；-999999 表示用 --wuji_hand_sim_policy_epoch）")
    p.add_argument("--wuji_hand_sim_use_fingertips5", action="store_true", help="sim 端 model 输入用 5 指尖（默认启用）")
    p.set_defaults(wuji_hand_sim_use_fingertips5=True)
    p.add_argument("--wuji_hand_sim_clamp_min", type=float, default=-1.5, help="sim 端 model 输出限幅最小值")
    p.add_argument("--wuji_hand_sim_clamp_max", type=float, default=1.5, help="sim 端 model 输出限幅最大值")
    p.add_argument("--wuji_hand_sim_max_delta_per_step", type=float, default=0.08, help="sim 端 model 输出每步最大变化")
    p.add_argument(
        "--wuji_hand_sim_log_dir",
        type=str,
        default="/tmp",
        help="Wuji 手 sim 日志输出目录（默认 /tmp）。",
    )

    p.add_argument("--target_fps", type=float, default=0.0, help="If >0, rate-limit output loop to this FPS")
    p.add_argument("--print_every", type=int, default=120, help="Print status every N frames (<=0 disable)")
    p.add_argument("--dry_run", action="store_true", help="Run retarget but do not write Redis")
    p.add_argument("--debug_viz_stages", action="store_true", help="Visualize raw/xform/gmr body frames (matplotlib)")
    p.add_argument("--debug_viz_viewer_axes", action="store_true", help="Match vmc_fk_viewer axes (x, z-depth, y-up)")
    p.add_argument("--debug_dump_vmc_raw", action="store_true", help="Print raw FK joints from vmc_viewer_fk")
    p.add_argument("--debug_viz_only_raw", action="store_true", help="Only visualize raw stage (matplotlib)")
    p.add_argument("--debug_viz_every", type=int, default=10, help="Update debug viz every N frames")
    p.add_argument("--debug_viz_scale", type=float, default=1.0, help="Scale for debug viz")
    p.add_argument("--swap_lr_body", action="store_true", help="Swap left/right body joints before retarget")
    p.add_argument("--body_mirror_x", action="store_true", help="Mirror body X axis before retarget")
    p.add_argument("--body_mirror_y", action="store_true", help="Mirror body Y axis before retarget")
    p.add_argument("--body_mirror_z", action="store_true", help="Mirror body Z axis before retarget")
    p.add_argument("--body_rot_x_180", action="store_true", help="Rotate body frame 180deg about X before retarget")
    p.add_argument("--body_rot_y_180", action="store_true", help="Rotate body frame 180deg about Y before retarget")
    p.add_argument("--body_rot_z_180", action="store_true", help="Rotate body frame 180deg about Z before retarget")
    p.add_argument("--csv_global_roll_deg", type=float, default=0.0, help="Global roll (deg) applied after --csv_apply_bvh_rotation")
    p.add_argument("--csv_global_pitch_deg", type=float, default=0.0, help="Global pitch (deg) applied after --csv_apply_bvh_rotation")
    p.add_argument("--csv_global_yaw_deg", type=float, default=0.0, help="Global yaw (deg) applied after --csv_apply_bvh_rotation")
    p.add_argument("--csv_global_rot_mode", type=str, default="world", help="Global rot mode after csv: world|basis")
    p.add_argument("--debug_dump_lr", action="store_true", help="Print L/R joint positions at each stage")
    p.add_argument("--debug_dump_every", type=int, default=30, help="Print debug dump every N frames")
    p.add_argument(
        "--safe_idle_pose_id",
        type=str,
        default="0",
        help=(
            "When k disables sending (send_enabled=False), publish this safe idle pose preset id (35D mimic_obs). "
            "支持单个数字（如 2）或列表（如 1,2）。当传列表时："
            "进入 default 会按序列依次平滑（例：1->2）；返回跟踪会先过渡到倒数第 1 个（例：2->1），再平滑回到正常跟踪。"
        ),
    )

    p.add_argument("--keyboard_toggle_send", action="store_true", help="Enable keyboard toggle (k/p)")
    p.add_argument("--toggle_send_key", type=str, default="k", help="Key to toggle send_enabled")
    p.add_argument("--hold_position_key", type=str, default="p", help="Key to toggle hold_position")
    p.add_argument(
        "--keyboard_backend",
        choices=["stdin", "evdev", "both"],
        default="stdin",
        help="键盘输入后端：stdin(需要终端在前台) / evdev(全局热键，不依赖前台，但需要 /dev/input 权限) / both(两者同时启用)",
    )
    p.add_argument("--evdev_device", type=str, default="auto", help="evdev 设备路径，如 /dev/input/event3 或 /dev/input/by-id/...；auto 尝试自动选择")
    p.add_argument("--evdev_grab", action="store_true", help="evdev grab（可能影响系统其他程序收到按键，谨慎使用）")

    # Smooth ramp when entering/exiting k/p modes and when exiting program
    p.add_argument(
        "--toggle_ramp_seconds",
        type=float,
        default=0.0,
        help="When k/p toggles change state, smoothly interpolate published action_body/action_neck for this duration (seconds). 0 disables.",
    )
    p.add_argument(
        "--exit_ramp_seconds",
        type=float,
        default=0.0,
        help="When exiting (q), smoothly ramp to safe idle for this duration (seconds) before quitting. 0 disables.",
    )
    p.add_argument(
        "--ramp_ease",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="Easing curve for ramp interpolation (default: cosine).",
    )
    p.add_argument(
        "--start_ramp_seconds",
        type=float,
        default=0.0,
        help="On startup (before any k/p toggles), smoothly ramp from safe idle to the live retarget action over this duration (seconds). 0 disables.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # imports that depend on repo_root
    from deploy_real.pose_csv_loader import (
        apply_bvh_like_coordinate_transform,
        apply_geo_to_bvh_official,
        gmr_rename_and_footmod,
    )
    from general_motion_retargeting import GeneralMotionRetargeting as GMR  # type: ignore
    from general_motion_retargeting import human_head_to_robot_neck  # type: ignore
    from data_utils.params import DEFAULT_MIMIC_OBS
    from data_utils.fps_monitor import FPSMonitor

    sdk = _import_xdmocap_sdk()
    MocapData = sdk["MocapData"]
    udp_open = sdk["udp_open"]
    udp_is_open = sdk["udp_is_open"]
    udp_close = sdk["udp_close"]
    udp_remove = sdk["udp_remove"]
    udp_send_request_connect = sdk["udp_send_request_connect"]
    udp_set_position_in_initial_tpose = sdk["udp_set_position_in_initial_tpose"]
    udp_recv_mocap_data = sdk["udp_recv_mocap_data"]
    length_body = int(sdk["LENGTH_BODY"])
    length_hand = int(sdk["LENGTH_HAND"])
    joint_names = [str(x) for x in sdk["NAMES_JOINT_BODY"]]
    joint_names_lhand = [str(x) for x in sdk["NAMES_JOINT_HAND_LEFT"]]
    joint_names_rhand = [str(x) for x in sdk["NAMES_JOINT_HAND_RIGHT"]]

    body_source = str(args.body_source).strip().lower()
    if body_source not in ["xdmocap", "vmc"]:
        print(f"❌ 不支持的 body_source: {body_source}")
        return 2
    vmc_receiver: Optional[VmcReceiver] = None
    vmc_viewer_fk = None
    vmc_viewer_seq = 0
    vmc_parents: dict[str, Optional[str]] = {}
    vmc_offsets: dict[str, np.ndarray] = {}
    vmc_bvh_to_vmc: dict[str, str] = {}
    vmc_root_pos: Optional[np.ndarray] = None
    vmc_bvh_axis_m: Optional[np.ndarray] = None
    vmc_axis_cycle = None
    vmc_axis_cycle_idx = 0
    vmc_axis_cycle_t0 = time.time()
    vmc_swap_lr = False
    if body_source == "vmc":
        invert_zw = True
        if bool(args.vmc_no_invert_zw):
            invert_zw = False
        elif bool(args.vmc_invert_zw):
            invert_zw = True
        if bool(args.vmc_use_viewer_fk):
            from deploy_real import vmc_fk_viewer as _vmc_viewer  # type: ignore
            # Align viewer FK skeleton with current args
            if str(args.vmc_fk_skeleton) == "bvh" or bool(args.vmc_bvh_lengths_only):
                bvh_parents, bvh_offsets = _parse_bvh_offsets(str(args.vmc_bvh_path))
                if bool(args.vmc_bvh_lengths_only):
                    _vmc_viewer.STD_SKELETON = dict(_vmc_viewer.STD_SKELETON_BASE)
                    for bone, (parent, off_std) in list(_vmc_viewer.STD_SKELETON.items()):
                        if bone not in bvh_offsets:
                            continue
                        off_bvh = np.array(bvh_offsets[bone], dtype=float)
                        len_bvh = float(np.linalg.norm(off_bvh)) * float(args.vmc_bvh_scale)
                        dir_std = np.array(off_std, dtype=float)
                        n = float(np.linalg.norm(dir_std))
                        if n < 1e-8:
                            continue
                        dir_std = dir_std / n
                        _vmc_viewer.STD_SKELETON[bone] = (
                            parent,
                            (dir_std[0] * len_bvh, dir_std[1] * len_bvh, dir_std[2] * len_bvh),
                        )
                else:
                    _vmc_viewer.STD_SKELETON = {}
                    for name, parent in bvh_parents.items():
                        off = bvh_offsets.get(name, np.array([0.0, 0.0, 0.0], dtype=float))
                        _vmc_viewer.STD_SKELETON[name] = (
                            parent,
                            (
                                float(off[0]) * float(args.vmc_bvh_scale),
                                float(off[1]) * float(args.vmc_bvh_scale),
                                float(off[2]) * float(args.vmc_bvh_scale),
                            ),
                        )
            vmc_viewer_fk = _vmc_viewer.VMCFKReceiver(str(args.vmc_ip), int(args.vmc_port))
            # Rebuild name_map/raw_rots to match current skeleton (same as viewer main)
            vmc_viewer_fk.raw_rots = {}
            vmc_viewer_fk.name_map = {}
            for bone in _vmc_viewer.STD_SKELETON:
                vmc_viewer_fk.raw_rots[bone] = np.array([0.0, 0.0, 0.0, 1.0])
            vmc_to_bvh = _vmc_viewer._build_vmc_to_bvh_map(set(_vmc_viewer.STD_SKELETON.keys()))
            for vmc_name, bvh_name in vmc_to_bvh.items():
                vmc_viewer_fk.name_map[_vmc_viewer._normalize_name(vmc_name)] = bvh_name
            vmc_viewer_fk.rot_mode = str(args.vmc_rot_mode)
            vmc_viewer_fk.invert_vmc_zw = bool(invert_zw)
            if str(args.vmc_viewer_bone_axis_override).strip():
                overrides = {}
                items = [s.strip() for s in str(args.vmc_viewer_bone_axis_override).split(";") if s.strip()]
                for item in items:
                    if ":" not in item:
                        continue
                    bone_name, cfg_str = item.split(":", 1)
                    bone_name = bone_name.strip()
                    cfg = {"swap": "xyz", "mirror_x": False, "mirror_y": False, "mirror_z": False}
                    parts = [p.strip() for p in cfg_str.split(",") if p.strip()]
                    for p in parts:
                        if p.startswith("swap="):
                            cfg["swap"] = p.split("=", 1)[1].strip()
                        elif p.startswith("flip="):
                            flips = p.split("=", 1)[1].strip()
                            cfg["mirror_x"] = "x" in flips
                            cfg["mirror_y"] = "y" in flips
                            cfg["mirror_z"] = "z" in flips
                    overrides[bone_name] = cfg
                vmc_viewer_fk.bone_axis_override = overrides
            vmc_viewer_fk.start()
        else:
            vmc_receiver = VmcReceiver(str(args.vmc_ip), int(args.vmc_port), invert_zw=invert_zw)
        vmc_swap_lr = bool(args.vmc_swap_lr)
        if bool(args.vmc_use_fk):
            if str(args.vmc_fk_skeleton) == "std":
                vmc_parents, vmc_offsets, vmc_root_pos = _std_fk_skeleton()
                if bool(args.vmc_bvh_lengths_only):
                    bvh_parents, bvh_offsets = _parse_bvh_offsets(str(args.vmc_bvh_path))
                    bvh_to_std = {
                        "HIP": "Hips",
                        "Hips": "Hips",
                        "WAIST": "Spine",
                        "Spine": "Spine",
                        "CHEST": "Chest",
                        "Chest": "Chest",
                        "UPPER_CHEST": "UpperChest",
                        "UpperChest": "UpperChest",
                        "NECK": "Neck",
                        "Neck": "Neck",
                        "HEAD": "Head",
                        "Head": "Head",
                        "LEFT_UPPER_SHOULDER": "LeftShoulder",
                        "LEFT_SHOULDER": "LeftShoulder",
                        "LeftShoulder": "LeftShoulder",
                        "LEFT_UPPER_ARM": "LeftUpperArm",
                        "LeftUpperArm": "LeftUpperArm",
                        "LEFT_LOWER_ARM": "LeftLowerArm",
                        "LeftLowerArm": "LeftLowerArm",
                        "LEFT_HAND": "LeftHand",
                        "LeftHand": "LeftHand",
                        "RIGHT_UPPER_SHOULDER": "RightShoulder",
                        "RIGHT_SHOULDER": "RightShoulder",
                        "RightShoulder": "RightShoulder",
                        "RIGHT_UPPER_ARM": "RightUpperArm",
                        "RightUpperArm": "RightUpperArm",
                        "RIGHT_LOWER_ARM": "RightLowerArm",
                        "RightLowerArm": "RightLowerArm",
                        "RIGHT_HAND": "RightHand",
                        "RightHand": "RightHand",
                        "LEFT_HIP": "Hips",
                        "LEFT_UPPER_LEG": "LeftUpperLeg",
                        "LeftUpperLeg": "LeftUpperLeg",
                        "LEFT_LOWER_LEG": "LeftLowerLeg",
                        "LeftLowerLeg": "LeftLowerLeg",
                        "LEFT_FOOT": "LeftFoot",
                        "LeftFoot": "LeftFoot",
                        "RIGHT_HIP": "Hips",
                        "RIGHT_UPPER_LEG": "RightUpperLeg",
                        "RightUpperLeg": "RightUpperLeg",
                        "RIGHT_LOWER_LEG": "RightLowerLeg",
                        "RightLowerLeg": "RightLowerLeg",
                        "RIGHT_FOOT": "RightFoot",
                        "RightFoot": "RightFoot",
                    }
                    for bvh_name, off_bvh in bvh_offsets.items():
                        if bvh_name not in bvh_to_std:
                            continue
                        bone = bvh_to_std[bvh_name]
                        if bone not in vmc_offsets:
                            continue
                        off_std = np.array(vmc_offsets[bone], dtype=float)
                        n = float(np.linalg.norm(off_std))
                        if n < 1e-8:
                            continue
                        dir_std = off_std / n
                        len_bvh = float(np.linalg.norm(off_bvh)) * float(args.vmc_bvh_scale)
                        vmc_offsets[bone] = dir_std * len_bvh
            else:
                vmc_parents, vmc_offsets = _parse_bvh_offsets(str(args.vmc_bvh_path))
                if bool(args.vmc_bvh_axis_auto):
                    swaps = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
                    flips = ["", "x", "y", "z", "xy", "xz", "yz", "xyz"]
                    vmc_axis_cycle = [(s, f) for s in swaps for f in flips]
                    vmc_axis_cycle_idx = 0
                    vmc_axis_cycle_t0 = time.time()
                    s0, f0 = vmc_axis_cycle[0]
                    vmc_bvh_axis_m = _axis_swap_flip_matrix(s0, f0)
                else:
                    vmc_bvh_axis_m = _axis_swap_flip_matrix(str(args.vmc_bvh_axis_swap), str(args.vmc_bvh_axis_flip))
            # BVH/STD -> VMC bone name mapping
            vmc_bvh_to_vmc = {
                "HIP": "Hips",
                "Hips": "Hips",
                "WAIST": "Spine",
                "Spine": "Spine",
                "CHEST": "Chest",
                "Chest": "Chest",
                "UPPER_CHEST": "UpperChest",
                "UpperChest": "UpperChest",
                "NECK": "Neck",
                "Neck": "Neck",
                "HEAD": "Head",
                "Head": "Head",
                "LEFT_UPPER_SHOULDER": "LeftShoulder",
                "LEFT_SHOULDER": "LeftShoulder",
                "LeftShoulder": "LeftShoulder",
                "LEFT_UPPER_ARM": "LeftUpperArm",
                "LeftUpperArm": "LeftUpperArm",
                "LEFT_LOWER_ARM": "LeftLowerArm",
                "LeftLowerArm": "LeftLowerArm",
                "LEFT_HAND": "LeftHand",
                "LeftHand": "LeftHand",
                "RIGHT_UPPER_SHOULDER": "RightShoulder",
                "RIGHT_SHOULDER": "RightShoulder",
                "RightShoulder": "RightShoulder",
                "RIGHT_UPPER_ARM": "RightUpperArm",
                "RightUpperArm": "RightUpperArm",
                "RIGHT_LOWER_ARM": "RightLowerArm",
                "RightLowerArm": "RightLowerArm",
                "RIGHT_HAND": "RightHand",
                "RightHand": "RightHand",
                "LEFT_HIP": "Hips",
                "LEFT_UPPER_LEG": "LeftUpperLeg",
                "LeftUpperLeg": "LeftUpperLeg",
                "LEFT_LOWER_LEG": "LeftLowerLeg",
                "LeftLowerLeg": "LeftLowerLeg",
                "LEFT_FOOT": "LeftFoot",
                "LeftFoot": "LeftFoot",
                "RIGHT_HIP": "Hips",
                "RIGHT_UPPER_LEG": "RightUpperLeg",
                "RightUpperLeg": "RightUpperLeg",
                "RIGHT_LOWER_LEG": "RightLowerLeg",
                "RightLowerLeg": "RightLowerLeg",
                "RIGHT_FOOT": "RightFoot",
                "RightFoot": "RightFoot",
            }
            if vmc_swap_lr:
                for k, v in list(vmc_bvh_to_vmc.items()):
                    if "Left" in v:
                        vmc_bvh_to_vmc[k] = v.replace("Left", "Right")
                    elif "Right" in v:
                        vmc_bvh_to_vmc[k] = v.replace("Right", "Left")

    # Resolve body/hand sources (keep backward compatibility)
    dst_ip_body = str(args.dst_ip_body).strip() or str(args.dst_ip)
    dst_port_body = int(args.dst_port_body) if int(args.dst_port_body) > 0 else int(args.dst_port)
    mocap_index_body = int(args.mocap_index_body) if int(args.mocap_index_body) >= 0 else int(args.mocap_index)

    dst_ip_hand = str(args.dst_ip_hand).strip() or str(args.dst_ip)
    dst_port_hand = int(args.dst_port_hand) if int(args.dst_port_hand) > 0 else int(args.dst_port)
    mocap_index_hand = int(args.mocap_index_hand) if int(args.mocap_index_hand) >= 0 else int(args.mocap_index)

    split_sources = (dst_ip_body != dst_ip_hand) or (dst_port_body != dst_port_hand) or (mocap_index_body != mocap_index_hand)

    # Init Redis
    client = redis.Redis(host=args.redis_ip, port=6379, db=0, decode_responses=False)
    try:
        client.ping()
    except Exception as e:
        print(f"❌ Redis 连接失败: {e}")
        return 3

    robot_key = "unitree_g1_with_hands"
    robot_name = "unitree_g1"
    key_action_body = f"action_body_{robot_key}"
    key_action_hand_l = f"action_hand_left_{robot_key}"
    key_action_hand_r = f"action_hand_right_{robot_key}"
    key_action_neck = f"action_neck_{robot_key}"
    key_t_action = "t_action"
    key_ht_l = f"hand_tracking_left_{robot_key}"
    key_ht_r = f"hand_tracking_right_{robot_key}"
    key_bvh_l = f"hand_bvh_left_{robot_key}"
    key_bvh_r = f"hand_bvh_right_{robot_key}"
    key_wuji_mode_l = f"wuji_hand_mode_left_{robot_key}"
    key_wuji_mode_r = f"wuji_hand_mode_right_{robot_key}"

    default_hand_7 = np.zeros(7, dtype=float).tolist()
    default_neck_2 = [0.0, 0.0]
    # 按 k（send_enabled=False）时发送的“安全 idle” 35D mimic_obs
    safe_idle_pose_ids = _parse_safe_idle_pose_ids(getattr(args, "safe_idle_pose_id", "0"))
    safe_idle_body_seq_35: list[np.ndarray] = []
    for _pid in safe_idle_pose_ids:
        _v = SAFE_IDLE_BODY_35_PRESETS[_pid]
        if len(_v) != 35:
            raise ValueError(f"SAFE_IDLE_BODY_35_PRESETS[{_pid}] must have length 35, got {len(_v)}")
        safe_idle_body_seq_35.append(np.asarray(_v, dtype=float).reshape(-1))
    # 最终稳定 idle 姿态：序列最后一个（兼容单个数字）
    safe_idle_body_35 = safe_idle_body_seq_35[-1].copy()

    # Init GMR
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot="unitree_g1",
        actual_human_height=float(args.actual_human_height),
    )

    # ---------------------------------------------------------------------
    # Spawn Wuji hand sim viz processes (keep it here; bash stays simple)
    # ---------------------------------------------------------------------
    wuji_sim_procs: list[subprocess.Popen] = []

    def _spawn_wuji_hand_sim(side: str) -> None:
        side = str(side).strip().lower()
        if side not in ["left", "right"]:
            return
        repo_root = _repo_root()
        script = (repo_root / "deploy_real" / "server_wuji_hand_sim_redis.py").resolve()
        if not script.exists():
            print(f"⚠️ 找不到 Wuji 手 sim 脚本：{script}（跳过启动）")
            return
        log_dir = Path(str(args.wuji_hand_sim_log_dir)).expanduser()
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            log_dir = Path("/tmp")
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        log_file = log_dir / f"wuji_hand_sim_{side}.log"

        cmd = [
            sys.executable,
            str(script),
            "--hand_side",
            side,
            "--redis_ip",
            str(args.redis_ip),
            "--target_fps",
            str(float(args.wuji_hand_sim_target_fps)),
        ]
        # mode switch: model inference
        if bool(args.wuji_hand_sim_use_model):
            tag_default = str(args.wuji_hand_sim_policy_tag)
            epoch_default = int(args.wuji_hand_sim_policy_epoch)
            if side == "left":
                tag = str(args.wuji_hand_sim_policy_tag_left) if str(args.wuji_hand_sim_policy_tag_left) else tag_default
                epoch_override = int(args.wuji_hand_sim_policy_epoch_left)
            else:
                tag = str(args.wuji_hand_sim_policy_tag_right) if str(args.wuji_hand_sim_policy_tag_right) else tag_default
                epoch_override = int(args.wuji_hand_sim_policy_epoch_right)
            epoch = epoch_default if epoch_override == -999999 else epoch_override

            cmd += [
                "--use_model",
                "--policy_tag",
                str(tag),
                "--policy_epoch",
                str(int(epoch)),
                "--clamp_min",
                str(float(args.wuji_hand_sim_clamp_min)),
                "--clamp_max",
                str(float(args.wuji_hand_sim_clamp_max)),
                "--max_delta_per_step",
                str(float(args.wuji_hand_sim_max_delta_per_step)),
            ]
            if bool(args.wuji_hand_sim_use_fingertips5):
                cmd += ["--use_fingertips5"]
        try:
            # NOTE: viewer 需要 GUI；无 GUI 时会报错并退出（日志里可见）
            f = open(str(log_file), "w", buffering=1)
            p = subprocess.Popen(
                cmd,
                cwd=str((repo_root / "deploy_real").resolve()),
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            wuji_sim_procs.append(p)
            print(f"🖐️  Wuji {side} hand sim started (pid={p.pid}) log={log_file}")
            # quick health check: if it exits immediately, warn user to check log
            time.sleep(0.2)
            if p.poll() is not None:
                print(f"⚠️  Wuji {side} hand sim 已立即退出（exit={p.returncode}），请查看日志：{log_file}")
        except Exception as e:
            print(f"⚠️ 启动 Wuji {side} hand sim 失败：{e}")

    def _cleanup_wuji_hand_sim() -> None:
        if not wuji_sim_procs:
            return
        for p in wuji_sim_procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        t0 = time.time()
        while time.time() - t0 < 1.0:
            alive = [p for p in wuji_sim_procs if p.poll() is None]
            if not alive:
                break
            time.sleep(0.05)
        for p in wuji_sim_procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass

    if bool(args.wuji_hand_sim_viz) and str(args.wuji_hand_sim_sides).lower() != "none":
        sides = str(args.wuji_hand_sim_sides).lower()
        if sides in ["left", "both"]:
            _spawn_wuji_hand_sim("left")
        if sides in ["right", "both"]:
            _spawn_wuji_hand_sim("right")

    # Optional MuJoCo viewer + video recording (ported from xrobot teleop)
    viewer_ctx = None
    viewer = None
    mj = None
    mjv = None
    model = None
    data = None
    renderer = None
    video_writer = None
    video_filename = None
    if bool(args.viz):
        try:
            import mujoco as mj  # type: ignore
            import mujoco.viewer as mjv  # type: ignore
            import cv2  # type: ignore
            from scipy.spatial.transform import Rotation as R  # type: ignore
            from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, draw_frame  # type: ignore

            xml_file = str(ROBOT_XML_DICT["unitree_g1"])
            robot_base = str(ROBOT_BASE_DICT["unitree_g1"])
            model = mj.MjModel.from_xml_path(xml_file)
            data = mj.MjData(model)

            viewer_ctx = mjv.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False)
            viewer = viewer_ctx.__enter__()
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1

            if bool(args.record_video):
                # Use a fixed renderer size to keep implementation simple.
                width, height, fps = 640, 480, 30
                video_filename = f"xdmocap_teleop_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                if vw.isOpened():
                    video_writer = vw
                    renderer = mj.Renderer(model, height=height, width=width)
                    print(f"[Video] Recording enabled: {video_filename} ({fps}fps, {width}x{height})")
                else:
                    try:
                        vw.release()
                    except Exception:
                        pass
                    video_writer = None
                    renderer = None
                    print("[Video] WARNING: failed to initialize VideoWriter, skip recording")

            def _viz_update(qpos: np.ndarray, human_frame: Optional[Dict[str, Any]]) -> None:
                if viewer is None or model is None or data is None:
                    return
                # clear custom geometry
                if hasattr(viewer, "user_scn") and viewer.user_scn is not None:
                    viewer.user_scn.ngeom = 0
                # draw IK target frames
                if human_frame is not None:
                    for _, entry in retargeter.ik_match_table1.items():
                        body_name = entry[0]
                        if body_name not in human_frame:
                            continue
                        pos = np.asarray(human_frame[body_name][0], dtype=np.float32).reshape(3)
                        quat = np.asarray(human_frame[body_name][1], dtype=np.float32).reshape(4)  # wxyz
                        draw_frame(pos - retargeter.ground, R.from_quat(quat, scalar_first=True).as_matrix(), viewer, 0.1)
                # update sim
                data.qpos[:] = qpos.copy()
                mj.mj_forward(model, data)
                # camera follow robot base
                try:
                    base_pos = data.xpos[model.body(robot_base).id]
                    viewer.cam.lookat = base_pos
                    viewer.cam.distance = 3.0
                except Exception:
                    pass
                viewer.sync()
                # record video frame
                if renderer is not None and video_writer is not None:
                    try:
                        renderer.update_scene(data, camera=viewer.cam)
                        pixels = renderer.render()
                        frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame)
                    except Exception:
                        pass

        except Exception as e:
            print(f"⚠️ 可视化初始化失败（继续无 viz 运行）：{e}")
            viewer = None
            if viewer_ctx is not None:
                try:
                    viewer_ctx.__exit__(None, None, None)
                except Exception:
                    pass
            viewer_ctx = None
            model = None
            data = None
            renderer = None
            video_writer = None
            video_filename = None

    else:
        def _viz_update(qpos: np.ndarray, human_frame: Optional[Dict[str, Any]]) -> None:
            return

    # Debug visualization for body pipeline stages (matplotlib)
    debug_viz = None
    debug_axes = None
    if bool(args.debug_viz_stages):
        debug_viz, debug_axes = _make_debug_viz(bool(args.debug_viz_only_raw))

    # Init keyboard
    kb = KeyboardToggle(
        enable=bool(args.keyboard_toggle_send),
        toggle_send_key=str(args.toggle_send_key),
        hold_key=str(args.hold_position_key),
        hand_step=float(args.hand_step),
        backend=str(args.keyboard_backend),
        evdev_device=str(args.evdev_device),
        evdev_grab=bool(args.evdev_grab),
    )
    kb.start()

    smooth = SmoothFilter(enable=bool(args.smooth), window_size=int(args.smooth_window_size))
    fps_mon = FPSMonitor(
        enable_detailed_stats=bool(int(args.measure_fps)),
        quick_print_interval=100,
        detailed_print_interval=1000,
        expected_fps=float(args.target_fps) if float(args.target_fps) > 1e-6 else None,
        name="XDMocap Teleop Loop",
    )

    # Init SDK UDP (body + hand can be split)
    mocap_body = MocapData()
    mocap_hand = mocap_body if (mocap_index_hand == mocap_index_body) else MocapData()

    def _ensure_open(idx: int) -> bool:
        if udp_is_open(int(idx)):
            return True
        return bool(udp_open(int(idx), int(args.local_port)))

    use_xdmocap_body = (body_source == "xdmocap")
    if use_xdmocap_body:
        if not _ensure_open(int(mocap_index_body)):
            print("❌ 无法打开 SDK UDP 端口（body）")
            kb.stop()
            return 4
    if int(mocap_index_hand) != int(mocap_index_body) or (not use_xdmocap_body):
        if not _ensure_open(int(mocap_index_hand)):
            print("❌ 无法打开 SDK UDP 端口（hand）")
            kb.stop()
            return 4

    def _try_init_sender(idx: int, ip: str, port: int) -> None:
        try:
            if not udp_set_position_in_initial_tpose(
                int(idx),
                str(ip),
                int(port),
                int(args.world_space),
                INITIAL_POSITION_BODY,
                INITIAL_POSITION_HAND_RIGHT,
                INITIAL_POSITION_HAND_LEFT,
            ):
                print(f"⚠️ udp_set_position_in_initial_tpose 失败：idx={idx} {ip}:{port}")
        except Exception:
            print(f"⚠️ udp_set_position_in_initial_tpose 异常：idx={idx} {ip}:{port}")

        print(f"正在连接动捕服务器 idx={idx} {ip}:{port} ...")
        if not udp_send_request_connect(int(idx), str(ip), int(port)):
            print(f"❌ 无法连接动捕服务器（udp_send_request_connect failed）：idx={idx} {ip}:{port}")
        else:
            print(f"✅ 动捕连接成功：idx={idx} {ip}:{port}")

    if use_xdmocap_body:
        _try_init_sender(int(mocap_index_body), str(dst_ip_body), int(dst_port_body))
    if split_sources or (not use_xdmocap_body):
        _try_init_sender(int(mocap_index_hand), str(dst_ip_hand), int(dst_port_hand))

    rate: Optional[RateLimiter] = None
    if float(args.target_fps) > 1e-6:
        rate = RateLimiter(frequency=float(args.target_fps), warn=False)

    print("=" * 70)
    print("Teleop -> Redis")
    print("=" * 70)
    print(f"body_source     : {body_source}")
    if body_source == "vmc":
        print(f"vmc_listen      : {args.vmc_ip}:{int(args.vmc_port)} (timeout={float(args.vmc_timeout_s):.3f}s)")
        print(f"vmc_rot_mode    : {str(args.vmc_rot_mode)}  invert_zw={bool(not args.vmc_no_invert_zw)}  use_fk={bool(args.vmc_use_fk)}  fk_skel={str(args.vmc_fk_skeleton)}  bvh_lengths_only={bool(args.vmc_bvh_lengths_only)}  swap_lr={bool(args.vmc_swap_lr)}  use_viewer_fk={bool(args.vmc_use_viewer_fk)}")
        if bool(args.vmc_use_fk):
            if bool(args.vmc_bvh_axis_auto):
                print(f"vmc_bvh_path    : {str(args.vmc_bvh_path)}  scale={float(args.vmc_bvh_scale):.3f}  axis=auto  cycle_s={float(args.vmc_bvh_axis_cycle_s):.2f}")
            else:
                print(f"vmc_bvh_path    : {str(args.vmc_bvh_path)}  scale={float(args.vmc_bvh_scale):.3f}  axis={str(args.vmc_bvh_axis_swap)} flip={str(args.vmc_bvh_axis_flip)}")
    print(f"swap_lr_body    : {bool(args.swap_lr_body)}")
    print(f"body_mirror     : x={bool(args.body_mirror_x)} y={bool(args.body_mirror_y)} z={bool(args.body_mirror_z)}")
    if split_sources:
        print(f"dst(body): {dst_ip_body}:{dst_port_body} (mocap_index={mocap_index_body}, ws={args.world_space})")
        print(f"dst(hand): {dst_ip_hand}:{dst_port_hand} (mocap_index={mocap_index_hand}, ws={args.world_space})")
    else:
        print(f"dst    : {args.dst_ip}:{args.dst_port} (mocap_index={int(args.mocap_index)}, ws={args.world_space})")
    print(f"redis  : {args.redis_ip}:6379")
    print(f"format : {args.format} (src_human=bvh_{args.format})")
    print(f"geo2bvh_official : {bool(args.csv_geo_to_bvh_official)}")
    print(f"bvh_rotation     : {bool(args.csv_apply_bvh_rotation)}")
    print(f"hands            : {str(args.hands)}")
    print(f"safe_idle_pose_id : {str(getattr(args, 'safe_idle_pose_id', '0'))} (parsed={safe_idle_pose_ids})")
    # pretty-print hand_fk_end_site_scale
    _hs = args.hand_fk_end_site_scale
    if isinstance(_hs, (list, tuple, np.ndarray)):
        hs_str = "[" + ",".join([f"{float(x):.3f}" for x in np.asarray(_hs).reshape(-1).tolist()]) + "]"
    else:
        hs_str = f"{float(_hs):.3f}"
    print(f"hand_fk          : {bool(args.hand_fk)} (end_site_scale={hs_str})")
    print(f"offset_to_ground : {bool(args.offset_to_ground)}")
    print(f"toggle_ramp_seconds : {float(args.toggle_ramp_seconds):.3f}")
    print(f"exit_ramp_seconds   : {float(args.exit_ramp_seconds):.3f}")
    print(f"ramp_ease           : {str(args.ramp_ease)}")
    print(f"start_ramp_seconds  : {float(args.start_ramp_seconds):.3f}")
    if args.dry_run:
        print("dry_run: True (不会写 Redis)")
    print("")

    # Precompute transformed hand T-pose bones for FK (keep consistent with runtime coordinate transforms)
    fk_bone_left: Optional[np.ndarray] = None
    fk_bone_right: Optional[np.ndarray] = None
    if bool(args.hand_fk):
        try:
            def _transform_init_bone(bone_xyz_list: list[list[float]], prefix: str) -> np.ndarray:
                names = _hand_joint_order_names(prefix)
                fr: Dict[str, Any] = {}
                for n, p0 in zip(names, bone_xyz_list):
                    fr[n] = [np.asarray(p0, dtype=np.float32).reshape(3), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]
                # IMPORTANT: when --hand_no_csv_transform is set, hand FK should stay in raw SDK space.
                if (not bool(args.hand_no_csv_transform)) and bool(args.csv_geo_to_bvh_official):
                    fr = apply_geo_to_bvh_official(fr)
                if (not bool(args.hand_no_csv_transform)) and bool(args.csv_apply_bvh_rotation):
                    fr = apply_bvh_like_coordinate_transform(
                        fr,
                        pos_unit="m",
                        apply_rotation=True,
                        rot_mode=str(args.csv_bvh_rot_mode),
                        rot_tweak=str(args.csv_bvh_rot_tweak),
                        rot_tweak_order=str(args.csv_bvh_rot_tweak_order),
                        apply_pos_rotation=True if (not (bool(args.csv_bvh_apply_pos) or bool(args.csv_bvh_apply_quat))) else bool(args.csv_bvh_apply_pos),
                        apply_quat_rotation=True if (not (bool(args.csv_bvh_apply_pos) or bool(args.csv_bvh_apply_quat))) else bool(args.csv_bvh_apply_quat),
                    )
                return np.stack([np.asarray(fr[n][0], dtype=np.float32).reshape(3) for n in names], axis=0)

            fk_bone_left = _transform_init_bone(INITIAL_POSITION_HAND_LEFT, "Left")
            fk_bone_right = _transform_init_bone(INITIAL_POSITION_HAND_RIGHT, "Right")
        except Exception as e:
            print(f"⚠️ hand_fk 初始化失败（将回退不用 FK）：{e}")
            fk_bone_left = None
            fk_bone_right = None

    last_body_frame_index: int = -1
    last_hand_frame_index: int = -1
    last_hand_frame_time: Optional[float] = None
    last_fr_hand: Dict[str, Any] = {}
    last_qpos: Optional[np.ndarray] = None
    last_time: Optional[float] = None

    cached_action_body_35 = np.array(safe_idle_body_35, dtype=float)
    cached_action_hand_left_7 = np.array(default_hand_7, dtype=float)
    cached_action_hand_right_7 = np.array(default_hand_7, dtype=float)
    cached_action_neck_2 = np.array(default_neck_2, dtype=float)

    # Last published (for ramp start)
    last_pub_body_35 = cached_action_body_35.copy()
    last_pub_neck_2 = cached_action_neck_2.copy()

    # Hold-freeze target: when entering hold, freeze current published action so it doesn't drift
    hold_frozen_body_35: Optional[np.ndarray] = None
    hold_frozen_neck_2: Optional[np.ndarray] = None

    # Ramp state: when k/p state changes or exiting, smoothly blend to target over time.
    ramp_active = False
    ramp_t0: Optional[float] = None
    ramp_seconds: float = 0.0
    ramp_from_body_35 = cached_action_body_35.copy()
    ramp_from_neck_2 = cached_action_neck_2.copy()
    ramp_exit_mode = False  # if True, ramping to idle before quit

    # Multi-stage ramp plan (optional): used when --safe_idle_pose_id is a list like "1,2"
    ramp_plan_kinds: list[str] = []  # "fixed" or "live"
    ramp_plan_body: list[Optional[np.ndarray]] = []
    ramp_plan_neck: list[Optional[np.ndarray]] = []
    ramp_plan_force_send: list[Optional[bool]] = []
    ramp_plan_force_hold: list[Optional[bool]] = []
    ramp_plan_seconds: list[float] = []
    ramp_stage_idx: int = 0

    last_send_enabled: Optional[bool] = None
    last_hold_enabled: Optional[bool] = None

    # Startup ramp: smooth transition from safe idle -> live retarget output.
    if float(args.start_ramp_seconds) > 1e-6:
        ramp_active = True
        ramp_exit_mode = False
        ramp_t0 = time.time()
        ramp_seconds = float(args.start_ramp_seconds)
        # Start from safe idle (same as k-disabled target) to avoid jerky first frame.
        ramp_from_body_35 = np.asarray(safe_idle_body_35, dtype=float).reshape(-1).copy()
        ramp_from_neck_2 = np.asarray(default_neck_2, dtype=float).reshape(-1).copy()

    def _refresh_hand_cache() -> None:
        nonlocal last_hand_frame_index, last_hand_frame_time, last_fr_hand
        got_hand = udp_recv_mocap_data(int(mocap_index_hand), str(dst_ip_hand), int(dst_port_hand), mocap_hand)
        if got_hand and getattr(mocap_hand, "isUpdate", False):
            fi_h = int(getattr(mocap_hand, "frameIndex", -1))
            if fi_h != last_hand_frame_index:
                last_hand_frame_index = fi_h
                hands_mode = str(args.hands).lower()
                fr_tmp: Dict[str, Any] = {}
                if hands_mode in ["left", "both"]:
                    fr_tmp.update(_build_hand_frame_from_sdk(mocap_hand, joint_names_lhand, is_left=True, length_hand=length_hand))
                if hands_mode in ["right", "both"]:
                    fr_tmp.update(_build_hand_frame_from_sdk(mocap_hand, joint_names_rhand, is_left=False, length_hand=length_hand))
                last_fr_hand = fr_tmp
                last_hand_frame_time = time.time()

    step = 0
    try:
        while True:
            fr_body: Optional[Dict[str, Any]] = None
            frame_id: int = -1

            if body_source == "vmc":
                if vmc_receiver is None and vmc_viewer_fk is None:
                    print("❌ VMC 接收器未初始化")
                    break
                if vmc_axis_cycle is not None:
                    if (time.time() - vmc_axis_cycle_t0) >= float(args.vmc_bvh_axis_cycle_s):
                        vmc_axis_cycle_t0 = time.time()
                        vmc_axis_cycle_idx = (vmc_axis_cycle_idx + 1) % len(vmc_axis_cycle)
                        s, f = vmc_axis_cycle[vmc_axis_cycle_idx]
                        vmc_bvh_axis_m = _axis_swap_flip_matrix(s, f)
                        print(f"[axis_cycle] swap={s} flip={f}")
                        if bool(args.vmc_bvh_axis_cycle_once) and vmc_axis_cycle_idx == 0:
                            print("✅ BVH axis cycle finished (once).")
                            return 0
                if vmc_viewer_fk is not None:
                    vmc_viewer_fk.solve_fk()
                    bones = _frame_from_pos_rot(vmc_viewer_fk.computed_pos, vmc_viewer_fk.computed_rot)
                    if bool(args.debug_dump_vmc_raw) and (step % int(args.debug_viz_every) == 0):
                        dump_names = ["Hips", "Spine", "LeftUpperArm", "RightUpperArm", "LeftFoot", "RightFoot"]
                        print(f"[vmc_viewer_fk t={time.time():.3f}]", flush=True)
                        for n in dump_names:
                            p = vmc_viewer_fk.computed_pos.get(n)
                            if p is None:
                                continue
                            print(f"  {n}: {np.asarray(p, dtype=np.float32).round(4)}", flush=True)
                    vmc_viewer_seq += 1
                    seq = int(vmc_viewer_seq)
                    _ts = time.time()
                else:
                    bones, seq, _ts = vmc_receiver.snapshot(float(args.vmc_timeout_s))
                if bones is None or seq < 0:
                    _refresh_hand_cache()
                    time.sleep(0.0005)
                    if rate is not None:
                        rate.sleep()
                    fps_mon.tick()
                    continue
                if seq == last_body_frame_index:
                    _refresh_hand_cache()
                    time.sleep(0.0002)
                    if rate is not None:
                        rate.sleep()
                    fps_mon.tick()
                    continue
                last_body_frame_index = seq
                frame_id = int(seq)
                if bool(args.vmc_use_fk):
                    # IMPORTANT: when using viewer FK, `bones` is already the solved FK pose
                    # (pos + global rot). Do NOT run our FK again, otherwise it diverges from vmc_fk_viewer.py.
                    if vmc_viewer_fk is not None:
                        vmc_pose = bones
                        fr_body = _vmc_build_body_frame(vmc_pose, joint_names)
                    else:
                        if str(args.vmc_fk_skeleton) == "std":
                            raw_xyzw, _seq2, _ts2 = vmc_receiver.snapshot_raw_xyzw(float(args.vmc_timeout_s))
                            if raw_xyzw is None:
                                _refresh_hand_cache()
                                time.sleep(0.0005)
                                if rate is not None:
                                    rate.sleep()
                                fps_mon.tick()
                                continue
                            vmc_pose = _build_fk_from_vmc_std(raw_xyzw, str(args.vmc_rot_mode))
                            fr_body = _vmc_build_body_frame(vmc_pose, joint_names)
                        else:
                            fk_pos, fk_rot = _build_fk_from_vmc(
                                bones,
                                vmc_parents,
                                vmc_offsets,
                                vmc_bvh_to_vmc,
                                float(args.vmc_bvh_scale),
                                str(args.vmc_rot_mode),
                                vmc_root_pos,
                                vmc_bvh_axis_m,
                            )
                            vmc_pose = _fk_to_vmc_pose(fk_pos, fk_rot, vmc_bvh_to_vmc)
                            fr_body = _vmc_build_body_frame(vmc_pose, joint_names)
                else:
                    fr_body = _vmc_build_body_frame(bones, joint_names)
                _refresh_hand_cache()
            else:
                got_body = udp_recv_mocap_data(int(mocap_index_body), str(dst_ip_body), int(dst_port_body), mocap_body)
                if not got_body or (not getattr(mocap_body, "isUpdate", False)):
                    # still try to refresh hand
                    _refresh_hand_cache()
                    time.sleep(0.0005)
                    if rate is not None:
                        rate.sleep()
                    fps_mon.tick()
                    continue

            if body_source != "vmc":
                fi = int(getattr(mocap_body, "frameIndex", -1))
                if fi == last_body_frame_index:
                    # body not updated; still try to refresh hand cache
                    _refresh_hand_cache()
                    time.sleep(0.0002)
                    if rate is not None:
                        rate.sleep()
                    fps_mon.tick()
                    continue
                last_body_frame_index = fi
                frame_id = int(fi)

                # Refresh hand cache (only needed when hand source is split from body).
                if split_sources:
                    _refresh_hand_cache()

            if body_source == "vmc" and fr_body is None:
                time.sleep(0.0002)
                if rate is not None:
                    rate.sleep()
                fps_mon.tick()
                continue

            send_enabled, hold_enabled, exit_requested, hand_left_val, hand_right_val = kb.get_extended_state()
            raw_send_enabled = bool(send_enabled)
            raw_hold_enabled = bool(hold_enabled)

            # detect toggle transitions for ramp
            if last_send_enabled is None:
                last_send_enabled = bool(raw_send_enabled)
            if last_hold_enabled is None:
                last_hold_enabled = bool(raw_hold_enabled)
            prev_send_enabled = bool(last_send_enabled)
            prev_hold_enabled = bool(last_hold_enabled)
            toggled = (bool(raw_send_enabled) != prev_send_enabled) or (bool(raw_hold_enabled) != prev_hold_enabled)

            # Entering/leaving hold: freeze/unfreeze target
            if (not prev_hold_enabled) and bool(raw_hold_enabled):
                hold_frozen_body_35 = last_pub_body_35.copy()
                hold_frozen_neck_2 = last_pub_neck_2.copy()
            if prev_hold_enabled and (not bool(raw_hold_enabled)):
                hold_frozen_body_35 = None
                hold_frozen_neck_2 = None

            # Exit handling: optionally ramp to safe idle before quitting
            if exit_requested and (not ramp_exit_mode) and float(args.exit_ramp_seconds) > 1e-6:
                ramp_active = True
                ramp_exit_mode = True
                ramp_t0 = time.time()
                ramp_seconds = float(args.exit_ramp_seconds)
                ramp_from_body_35 = last_pub_body_35.copy()
                ramp_from_neck_2 = last_pub_neck_2.copy()
                # During exit ramp: treat as disabled (idle + wuji default)
                raw_send_enabled = False
                raw_hold_enabled = False
                # exit ramp: keep simple (no multi-stage)
                ramp_plan_kinds = []
                ramp_plan_body = []
                ramp_plan_neck = []
                ramp_plan_force_send = []
                ramp_plan_force_hold = []
                ramp_plan_seconds = []
                ramp_stage_idx = 0

            # k/p toggle ramp: start on state change (both enter/exit)
            if toggled and (not ramp_exit_mode) and float(args.toggle_ramp_seconds) > 1e-6:
                ramp_active = True
                ramp_t0 = time.time()
                ramp_seconds = float(args.toggle_ramp_seconds)
                ramp_from_body_35 = last_pub_body_35.copy()
                ramp_from_neck_2 = last_pub_neck_2.copy()
                # Optional multi-stage plan for send_enabled toggle and safe_idle list (e.g. "1,2")
                ramp_plan_kinds = []
                ramp_plan_body = []
                ramp_plan_neck = []
                ramp_plan_force_send = []
                ramp_plan_force_hold = []
                ramp_plan_seconds = []
                ramp_stage_idx = 0
                secs_total = float(args.toggle_ramp_seconds)
                if (bool(raw_send_enabled) != bool(prev_send_enabled)) and (len(safe_idle_pose_ids) >= 2) and secs_total > 1e-6:
                    if not bool(raw_send_enabled):
                        # entering default: ... -> 1 -> 2 (each stage fixed, forced disabled)
                        n = int(len(safe_idle_body_seq_35))
                        per = secs_total / float(max(1, n))
                        for i in range(n):
                            ramp_plan_kinds.append("fixed")
                            ramp_plan_body.append(np.asarray(safe_idle_body_seq_35[i], dtype=float).reshape(-1).copy())
                            ramp_plan_neck.append(np.asarray(default_neck_2, dtype=float).reshape(-1))
                            ramp_plan_force_send.append(False)
                            ramp_plan_force_hold.append(False)
                            ramp_plan_seconds.append(float(per))
                        ramp_seconds = float(ramp_plan_seconds[0])
                    else:
                        # leaving default: 2 -> 1 -> follow
                        stages = [
                            ("fixed", np.asarray(safe_idle_body_seq_35[-2], dtype=float).reshape(-1).copy(), np.asarray(default_neck_2, dtype=float).reshape(-1), False, False),
                            ("live", None, None, True, None),
                        ]
                        per = secs_total / float(len(stages))
                        for kind, body, neck, fsend, fhold in stages:
                            ramp_plan_kinds.append(str(kind))
                            ramp_plan_body.append(body)
                            ramp_plan_neck.append(neck)
                            ramp_plan_force_send.append(fsend)
                            ramp_plan_force_hold.append(fhold)
                            ramp_plan_seconds.append(float(per))
                        ramp_seconds = float(ramp_plan_seconds[0])

                last_send_enabled = bool(raw_send_enabled)
                last_hold_enabled = bool(raw_hold_enabled)
            else:
                last_send_enabled = bool(raw_send_enabled)
                last_hold_enabled = bool(raw_hold_enabled)

            # Apply ramp-stage override to effective k/p state
            send_enabled = bool(raw_send_enabled)
            hold_enabled = bool(raw_hold_enabled)
            if ramp_active and (not ramp_exit_mode) and ramp_plan_kinds and (0 <= ramp_stage_idx < len(ramp_plan_kinds)):
                fsend = ramp_plan_force_send[ramp_stage_idx]
                fhold = ramp_plan_force_hold[ramp_stage_idx]
                if fsend is not None:
                    send_enabled = bool(fsend)
                if fhold is not None:
                    hold_enabled = bool(fhold)
            if not bool(send_enabled):
                hold_enabled = False
            now = time.time()
            if last_time is None:
                dt = 1.0 / 60.0
            else:
                dt = max(1e-4, float(now - last_time))
            last_time = now

            # -----------------------------
            # Fast path for hold/disabled (do NOT update cached_action_* from mocap in these modes)
            # -----------------------------
            if hold_enabled or (not send_enabled):
                if hold_enabled:
                    target_body_35 = (hold_frozen_body_35 if hold_frozen_body_35 is not None else last_pub_body_35).copy()
                    target_neck_2 = (hold_frozen_neck_2 if hold_frozen_neck_2 is not None else last_pub_neck_2).copy()
                    mode_l, mode_r = "hold", "hold"
                else:
                    # If multi-stage plan is active, allow intermediate idle pose (e.g. 1 before final 2).
                    if ramp_active and (not ramp_exit_mode) and ramp_plan_kinds and (0 <= ramp_stage_idx < len(ramp_plan_kinds)) and ramp_plan_kinds[ramp_stage_idx] == "fixed":
                        _b = ramp_plan_body[ramp_stage_idx]
                        _n = ramp_plan_neck[ramp_stage_idx]
                        target_body_35 = np.asarray(_b if _b is not None else safe_idle_body_35, dtype=float).reshape(-1)
                        target_neck_2 = np.asarray(_n if _n is not None else default_neck_2, dtype=float).reshape(-1)
                    else:
                        target_body_35 = np.asarray(safe_idle_body_35, dtype=float).reshape(-1)
                        target_neck_2 = np.asarray(default_neck_2, dtype=float).reshape(-1)
                    mode_l, mode_r = "default", "default"

                pub_body_35 = target_body_35
                pub_neck_2 = target_neck_2
                if ramp_active and (ramp_t0 is not None) and ramp_seconds > 1e-6:
                    alpha = (time.time() - float(ramp_t0)) / max(1e-6, float(ramp_seconds))
                    w = _ease(alpha, ease=str(args.ramp_ease))
                    pub_body_35 = (1.0 - w) * ramp_from_body_35 + w * target_body_35
                    pub_neck_2 = (1.0 - w) * ramp_from_neck_2 + w * target_neck_2
                    if alpha >= 1.0:
                        if ramp_exit_mode:
                            ramp_active = False
                            ramp_t0 = None
                            ramp_seconds = 0.0
                            # publish final idle once then quit
                            if not args.dry_run:
                                now_ms = _now_ms()
                                pipe = client.pipeline()
                                pipe.set(key_action_body, json.dumps(np.asarray(pub_body_35, dtype=float).reshape(-1).tolist()))
                                pipe.set(key_action_hand_l, json.dumps(cached_action_hand_left_7.reshape(-1).tolist()))
                                pipe.set(key_action_hand_r, json.dumps(cached_action_hand_right_7.reshape(-1).tolist()))
                                pipe.set(key_action_neck, json.dumps(np.asarray(pub_neck_2, dtype=float).reshape(-1).tolist()))
                                pipe.set(key_t_action, now_ms)
                                pipe.set(key_wuji_mode_l, "default")
                                pipe.set(key_wuji_mode_r, "default")
                                pipe.set(key_ht_l, json.dumps({"is_active": False, "timestamp": now_ms}))
                                pipe.set(key_ht_r, json.dumps({"is_active": False, "timestamp": now_ms}))
                                pipe.execute()
                            break
                        # Multi-stage progression (if configured)
                        if ramp_plan_kinds and (ramp_stage_idx < len(ramp_plan_kinds) - 1):
                            ramp_stage_idx += 1
                            ramp_t0 = time.time()
                            ramp_seconds = float(ramp_plan_seconds[ramp_stage_idx])
                            ramp_from_body_35 = np.asarray(pub_body_35, dtype=float).reshape(-1).copy()
                            ramp_from_neck_2 = np.asarray(pub_neck_2, dtype=float).reshape(-1).copy()
                        else:
                            ramp_active = False
                            ramp_t0 = None
                            ramp_seconds = 0.0
                            ramp_plan_kinds = []
                            ramp_plan_body = []
                            ramp_plan_neck = []
                            ramp_plan_force_send = []
                            ramp_plan_force_hold = []
                            ramp_plan_seconds = []
                            ramp_stage_idx = 0

                last_pub_body_35 = np.asarray(pub_body_35, dtype=float).reshape(-1).copy()
                last_pub_neck_2 = np.asarray(pub_neck_2, dtype=float).reshape(-1).copy()

                if not args.dry_run:
                    now_ms = _now_ms()
                    pipe = client.pipeline()
                    pipe.set(key_action_body, json.dumps(last_pub_body_35.reshape(-1).tolist()))
                    pipe.set(key_action_hand_l, json.dumps(cached_action_hand_left_7.reshape(-1).tolist()))
                    pipe.set(key_action_hand_r, json.dumps(cached_action_hand_right_7.reshape(-1).tolist()))
                    pipe.set(key_action_neck, json.dumps(last_pub_neck_2.reshape(-1).tolist()))
                    pipe.set(key_t_action, now_ms)
                    pipe.set(key_wuji_mode_l, mode_l)
                    pipe.set(key_wuji_mode_r, mode_r)
                    pipe.set(key_ht_l, json.dumps({"is_active": False, "timestamp": now_ms}))
                    pipe.set(key_ht_r, json.dumps({"is_active": False, "timestamp": now_ms}))
                    if bool(args.publish_bvh_hand):
                        pipe.set(key_bvh_l, json.dumps({"is_active": False, "timestamp": now_ms}))
                        pipe.set(key_bvh_r, json.dumps({"is_active": False, "timestamp": now_ms}))
                    pipe.execute()

                step += 1
                if rate is not None:
                    rate.sleep()
                fps_mon.tick()
                continue

            # -----------------------------
            # Normal path: compute target from mocap
            # -----------------------------
            target_body_35: np.ndarray = cached_action_body_35.copy()
            target_neck_2: np.ndarray = cached_action_neck_2.copy()
            mode_l: str = "follow"
            mode_r: str = "follow"
            ht_active: bool = True

            # 1) Raw frames (body from source, hands from SDK)
            if fr_body is None:
                fr_body = _build_body_frame_from_sdk(mocap_body, joint_names, length_body)
            fr_body_raw = _frame_copy(fr_body)
            hands_mode = str(args.hands).lower()

            # Hands:
            # - If NOT split_sources: hands come from the same mocap packet as body (old behavior).
            # - If split_sources: use cached hand frames + freshness timeout.
            if hands_mode == "none":
                fr_hand = {}
                hand_fresh = False
            elif not split_sources:
                fr_hand = {}
                if hands_mode in ["left", "both"]:
                    fr_hand.update(_build_hand_frame_from_sdk(mocap_body, joint_names_lhand, is_left=True, length_hand=length_hand))
                if hands_mode in ["right", "both"]:
                    fr_hand.update(_build_hand_frame_from_sdk(mocap_body, joint_names_rhand, is_left=False, length_hand=length_hand))
                # same packet => always fresh
                hand_fresh = True
                last_fr_hand = dict(fr_hand)
                last_hand_frame_time = time.time()
                last_hand_frame_index = int(getattr(mocap_body, "frameIndex", -1))
            else:
                fr_hand = dict(last_fr_hand)
                hand_fresh = False
                if last_hand_frame_time is not None:
                    hand_fresh = (time.time() - float(last_hand_frame_time)) <= float(args.hand_source_timeout_s)

            # 2) coordinate transforms (body always follows flags; hands can skip via --hand_no_csv_transform)
            if bool(args.csv_geo_to_bvh_official):
                fr_body = apply_geo_to_bvh_official(fr_body)
                if not bool(args.hand_no_csv_transform):
                    fr_hand = apply_geo_to_bvh_official(fr_hand)
            if bool(args.csv_apply_bvh_rotation):
                fr_body = apply_bvh_like_coordinate_transform(
                    fr_body,
                    pos_unit="m",
                    apply_rotation=True,
                    rot_mode=str(args.csv_bvh_rot_mode),
                    rot_tweak=str(args.csv_bvh_rot_tweak),
                    rot_tweak_order=str(args.csv_bvh_rot_tweak_order),
                    apply_pos_rotation=True if (not (bool(args.csv_bvh_apply_pos) or bool(args.csv_bvh_apply_quat))) else bool(args.csv_bvh_apply_pos),
                    apply_quat_rotation=True if (not (bool(args.csv_bvh_apply_pos) or bool(args.csv_bvh_apply_quat))) else bool(args.csv_bvh_apply_quat),
                )
                if not bool(args.hand_no_csv_transform):
                    fr_hand = apply_bvh_like_coordinate_transform(
                        fr_hand,
                        pos_unit="m",
                        apply_rotation=True,
                        rot_mode=str(args.csv_bvh_rot_mode),
                        rot_tweak=str(args.csv_bvh_rot_tweak),
                        rot_tweak_order=str(args.csv_bvh_rot_tweak_order),
                        apply_pos_rotation=True if (not (bool(args.csv_bvh_apply_pos) or bool(args.csv_bvh_apply_quat))) else bool(args.csv_bvh_apply_pos),
                        apply_quat_rotation=True if (not (bool(args.csv_bvh_apply_pos) or bool(args.csv_bvh_apply_quat))) else bool(args.csv_bvh_apply_quat),
                    )
                # extra single global rotation knob, after csv_apply_bvh_rotation
                if abs(float(args.csv_global_roll_deg)) > 1e-6 or abs(float(args.csv_global_pitch_deg)) > 1e-6 or abs(float(args.csv_global_yaw_deg)) > 1e-6:
                    Rw = _rpy_deg_to_rot(float(args.csv_global_roll_deg), float(args.csv_global_pitch_deg), float(args.csv_global_yaw_deg))
                    fr_body = _apply_global_rot_frame(fr_body, Rw, str(args.csv_global_rot_mode))
                    if not bool(args.hand_no_csv_transform):
                        fr_hand = _apply_global_rot_frame(fr_hand, Rw, str(args.csv_global_rot_mode))
            # proper rotations first (avoid reflection issues)
            if bool(args.body_rot_x_180) or bool(args.body_rot_y_180) or bool(args.body_rot_z_180):
                Rw = np.eye(3, dtype=np.float32)
                if bool(args.body_rot_x_180):
                    Rw = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32) @ Rw
                if bool(args.body_rot_y_180):
                    Rw = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32) @ Rw
                if bool(args.body_rot_z_180):
                    Rw = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32) @ Rw
                fr_body = _rotate_frame(fr_body, Rw)
            if bool(args.swap_lr_body):
                fr_body = _swap_lr_frame(fr_body)
            fr_body = _mirror_frame_axes(
                fr_body,
                mirror_x=bool(args.body_mirror_x),
                mirror_y=bool(args.body_mirror_y),
                mirror_z=bool(args.body_mirror_z),
            )
            fr_body_xform = _frame_copy(fr_body)

            # Combine (for hand processing/publishing). Note: body/hands may be in different spaces if hand_no_csv_transform=True.
            fr_all: Dict[str, Any] = {}
            fr_all.update(fr_body)
            fr_all.update(fr_hand)

            # 3) rename + FootMod (GMR retarget only needs body joints; keep it consistent)
            fr_gmr = gmr_rename_and_footmod(fr_body, fmt=str(args.format))

            # 4) IK retarget
            qpos = retargeter.retarget(fr_gmr, offset_to_ground=bool(args.offset_to_ground))

            # 5) mimic_obs (35D)
            if last_qpos is None:
                mimic = np.array(safe_idle_body_35, dtype=float)
            else:
                mimic = extract_mimic_obs_whole_body(qpos, last_qpos, dt=dt).astype(float)
            last_qpos = qpos.copy()
            sm = smooth.apply(mimic)
            if sm is not None:
                mimic = sm
            cached_action_body_35 = np.asarray(mimic, dtype=float).copy()

            # optional neck control
            if bool(args.control_neck):
                try:
                    neck_yaw, neck_pitch = human_head_to_robot_neck(fr_gmr)
                    s = float(args.neck_retarget_scale)
                    cached_action_neck_2 = np.array([float(neck_yaw) * s, float(neck_pitch) * s], dtype=float)
                except Exception:
                    cached_action_neck_2 = np.array(default_neck_2, dtype=float)
            else:
                cached_action_neck_2 = np.array(default_neck_2, dtype=float)

            # update target for normal path after computation
            if send_enabled and (not hold_enabled):
                target_body_35 = cached_action_body_35.copy()
                target_neck_2 = cached_action_neck_2.copy()
                mode_l, mode_r = "follow", "follow"
                ht_active = bool(hand_fresh)

            # optional Unitree gripper hand action (7D)
            if bool(args.control_gripper_hand_action):
                lh7, rh7 = _hand_pose_from_value(robot_name, hand_left_val, hand_right_val, pinch_mode=bool(args.pinch_mode))
                cached_action_hand_left_7 = np.asarray(lh7, dtype=float).reshape(-1)
                cached_action_hand_right_7 = np.asarray(rh7, dtype=float).reshape(-1)
            else:
                cached_action_hand_left_7 = np.array(default_hand_7, dtype=float)
                cached_action_hand_right_7 = np.array(default_hand_7, dtype=float)

            # -----------------------------
            # Apply ramp (if active) to body+neck
            # -----------------------------
            pub_body_35 = target_body_35
            pub_neck_2 = target_neck_2
            if ramp_active and (ramp_t0 is not None) and ramp_seconds > 1e-6:
                alpha = (time.time() - float(ramp_t0)) / max(1e-6, float(ramp_seconds))
                w = _ease(alpha, ease=str(args.ramp_ease))
                pub_body_35 = (1.0 - w) * ramp_from_body_35 + w * target_body_35
                pub_neck_2 = (1.0 - w) * ramp_from_neck_2 + w * target_neck_2
                if alpha >= 1.0:
                    if ramp_exit_mode:
                        ramp_active = False
                        ramp_t0 = None
                        ramp_seconds = 0.0
                        # after exit ramp finished, quit loop
                        if not args.dry_run:
                            now_ms = _now_ms()
                            pipe = client.pipeline()
                            pipe.set(key_action_body, json.dumps(pub_body_35.reshape(-1).tolist()))
                            pipe.set(key_action_hand_l, json.dumps(cached_action_hand_left_7.reshape(-1).tolist()))
                            pipe.set(key_action_hand_r, json.dumps(cached_action_hand_right_7.reshape(-1).tolist()))
                            pipe.set(key_action_neck, json.dumps(pub_neck_2.reshape(-1).tolist()))
                            pipe.set(key_t_action, now_ms)
                            pipe.set(key_wuji_mode_l, "default")
                            pipe.set(key_wuji_mode_r, "default")
                            pipe.set(key_ht_l, json.dumps({"is_active": False, "timestamp": now_ms}))
                            pipe.set(key_ht_r, json.dumps({"is_active": False, "timestamp": now_ms}))
                            pipe.execute()
                        break
                    # Multi-stage progression (if configured)
                    if ramp_plan_kinds and (ramp_stage_idx < len(ramp_plan_kinds) - 1):
                        ramp_stage_idx += 1
                        ramp_t0 = time.time()
                        ramp_seconds = float(ramp_plan_seconds[ramp_stage_idx])
                        ramp_from_body_35 = np.asarray(pub_body_35, dtype=float).reshape(-1).copy()
                        ramp_from_neck_2 = np.asarray(pub_neck_2, dtype=float).reshape(-1).copy()
                    else:
                        ramp_active = False
                        ramp_t0 = None
                        ramp_seconds = 0.0
                        ramp_plan_kinds = []
                        ramp_plan_body = []
                        ramp_plan_neck = []
                        ramp_plan_force_send = []
                        ramp_plan_force_hold = []
                        ramp_plan_seconds = []
                        ramp_stage_idx = 0

            # remember last published for next ramp start
            last_pub_body_35 = np.asarray(pub_body_35, dtype=float).reshape(-1).copy()
            last_pub_neck_2 = np.asarray(pub_neck_2, dtype=float).reshape(-1).copy()

            if not args.dry_run:
                now_ms = _now_ms()
                pipe = client.pipeline()
                pipe.set(key_action_body, json.dumps(last_pub_body_35.reshape(-1).tolist()))
                pipe.set(key_action_hand_l, json.dumps(cached_action_hand_left_7.reshape(-1).tolist()))
                pipe.set(key_action_hand_r, json.dumps(cached_action_hand_right_7.reshape(-1).tolist()))
                pipe.set(key_action_neck, json.dumps(last_pub_neck_2.reshape(-1).tolist()))
                pipe.set(key_t_action, now_ms)

                # Wuji 手控制：follow + hand_tracking_*
                if hands_mode == "none":
                    pipe.set(key_wuji_mode_l, "default")
                    pipe.set(key_wuji_mode_r, "default")
                    pipe.set(key_ht_l, json.dumps({"is_active": False, "timestamp": now_ms}))
                    pipe.set(key_ht_r, json.dumps({"is_active": False, "timestamp": now_ms}))
                    if bool(args.publish_bvh_hand):
                        pipe.set(key_bvh_l, json.dumps({"is_active": False, "timestamp": now_ms}))
                        pipe.set(key_bvh_r, json.dumps({"is_active": False, "timestamp": now_ms}))
                else:
                    pipe.set(key_wuji_mode_l, mode_l)
                    pipe.set(key_wuji_mode_r, mode_r)
                    if hands_mode in ["left", "both"]:
                        pos_override = None
                        tip_override = None
                        if bool(args.hand_fk) and (fk_bone_left is not None):
                            try:
                                prefix = "Left"
                                names = _hand_joint_order_names(prefix)
                                q20 = []
                                for n in names:
                                    v = fr_hand.get(n, None)
                                    if not (isinstance(v, (list, tuple)) and len(v) >= 2):
                                        raise KeyError(f"missing joint in frame: {n}")
                                    q20.append(_safe_quat_wxyz(np.asarray(v[1], dtype=np.float32)))
                                q20 = np.stack(q20, axis=0)
                                root_pos = np.asarray(fr_hand[f"{prefix}Hand"][0], dtype=np.float32).reshape(3)
                                pos20, pos_end5 = _fk_hand_positions_with_end_sites(
                                    q20,
                                    root_pos=root_pos,
                                    bone_init_pos=fk_bone_left,
                                    end_site_scale=args.hand_fk_end_site_scale,
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
                        # Only active when hand source is fresh
                        ht_l = {"is_active": bool(ht_active), "timestamp": now_ms}
                        if bool(ht_active):
                            ht_l.update(_hand_to_tracking26(fr_hand, "left", pos_override=pos_override, tip_override=tip_override))
                        pipe.set(key_ht_l, json.dumps(ht_l))
                        if bool(args.publish_bvh_hand):
                            # Publish BVH-style joints (positions+identity quat) for visualization
                            pfx = "Left"
                            bvh_names = [
                                f"{pfx}Hand",
                                f"{pfx}ThumbFinger", f"{pfx}ThumbFinger1", f"{pfx}ThumbFinger2",
                                f"{pfx}IndexFinger", f"{pfx}IndexFinger1", f"{pfx}IndexFinger2", f"{pfx}IndexFinger3",
                                f"{pfx}MiddleFinger", f"{pfx}MiddleFinger1", f"{pfx}MiddleFinger2", f"{pfx}MiddleFinger3",
                                f"{pfx}RingFinger", f"{pfx}RingFinger1", f"{pfx}RingFinger2", f"{pfx}RingFinger3",
                                f"{pfx}PinkyFinger", f"{pfx}PinkyFinger1", f"{pfx}PinkyFinger2", f"{pfx}PinkyFinger3",
                            ]
                            bvh_payload = {"is_active": True, "timestamp": now_ms}
                            for n in bvh_names:
                                v = pos_override.get(n, None) if pos_override is not None else None
                                if v is None:
                                    vv = fr_hand.get(n, None)
                                    if isinstance(vv, (list, tuple)) and len(vv) >= 1:
                                        v = np.asarray(vv[0], dtype=np.float32).reshape(3)
                                if v is None:
                                    continue
                                bvh_payload[n] = [np.asarray(v, dtype=np.float32).reshape(3).tolist(), [1.0, 0.0, 0.0, 0.0]]
                            pipe.set(key_bvh_l, json.dumps(bvh_payload))
                    else:
                        pipe.set(key_ht_l, json.dumps({"is_active": False, "timestamp": now_ms}))
                        if bool(args.publish_bvh_hand):
                            pipe.set(key_bvh_l, json.dumps({"is_active": False, "timestamp": now_ms}))
                    if hands_mode in ["right", "both"]:
                        pos_override = None
                        tip_override = None
                        if bool(args.hand_fk) and (fk_bone_right is not None):
                            try:
                                prefix = "Right"
                                names = _hand_joint_order_names(prefix)
                                q20 = []
                                for n in names:
                                    v = fr_hand.get(n, None)
                                    if not (isinstance(v, (list, tuple)) and len(v) >= 2):
                                        raise KeyError(f"missing joint in frame: {n}")
                                    q20.append(_safe_quat_wxyz(np.asarray(v[1], dtype=np.float32)))
                                q20 = np.stack(q20, axis=0)
                                root_pos = np.asarray(fr_hand[f"{prefix}Hand"][0], dtype=np.float32).reshape(3)
                                pos20, pos_end5 = _fk_hand_positions_with_end_sites(
                                    q20,
                                    root_pos=root_pos,
                                    bone_init_pos=fk_bone_right,
                                    end_site_scale=args.hand_fk_end_site_scale,
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
                        ht_r = {"is_active": bool(ht_active), "timestamp": now_ms}
                        if bool(ht_active):
                            ht_r.update(_hand_to_tracking26(fr_hand, "right", pos_override=pos_override, tip_override=tip_override))
                        pipe.set(key_ht_r, json.dumps(ht_r))
                        if bool(args.publish_bvh_hand):
                            pfx = "Right"
                            bvh_names = [
                                f"{pfx}Hand",
                                f"{pfx}ThumbFinger", f"{pfx}ThumbFinger1", f"{pfx}ThumbFinger2",
                                f"{pfx}IndexFinger", f"{pfx}IndexFinger1", f"{pfx}IndexFinger2", f"{pfx}IndexFinger3",
                                f"{pfx}MiddleFinger", f"{pfx}MiddleFinger1", f"{pfx}MiddleFinger2", f"{pfx}MiddleFinger3",
                                f"{pfx}RingFinger", f"{pfx}RingFinger1", f"{pfx}RingFinger2", f"{pfx}RingFinger3",
                                f"{pfx}PinkyFinger", f"{pfx}PinkyFinger1", f"{pfx}PinkyFinger2", f"{pfx}PinkyFinger3",
                            ]
                            bvh_payload = {"is_active": True, "timestamp": now_ms}
                            for n in bvh_names:
                                v = pos_override.get(n, None) if pos_override is not None else None
                                if v is None:
                                    vv = fr_hand.get(n, None)
                                    if isinstance(vv, (list, tuple)) and len(vv) >= 1:
                                        v = np.asarray(vv[0], dtype=np.float32).reshape(3)
                                if v is None:
                                    continue
                                bvh_payload[n] = [np.asarray(v, dtype=np.float32).reshape(3).tolist(), [1.0, 0.0, 0.0, 0.0]]
                            pipe.set(key_bvh_r, json.dumps(bvh_payload))
                    else:
                        pipe.set(key_ht_r, json.dumps({"is_active": False, "timestamp": now_ms}))
                        if bool(args.publish_bvh_hand):
                            pipe.set(key_bvh_r, json.dumps({"is_active": False, "timestamp": now_ms}))
            pipe.execute()

            # Visualization update (if enabled)
            _viz_update(qpos, fr_gmr)
            if debug_axes is not None and int(args.debug_viz_every) > 0 and (step % int(args.debug_viz_every) == 0):
                try:
                    _viz_body_frame(debug_axes[0], fr_body_raw, float(args.debug_viz_scale), bool(args.debug_viz_viewer_axes))
                    if bool(args.debug_viz_only_raw):
                        # single-axis mode: nothing else to draw
                        pass
                    else:
                        _viz_body_frame(debug_axes[1], fr_body_xform, float(args.debug_viz_scale), bool(args.debug_viz_viewer_axes))
                        _viz_body_frame(debug_axes[2], fr_gmr, float(args.debug_viz_scale), bool(args.debug_viz_viewer_axes))
                    import matplotlib.pyplot as plt  # type: ignore
                    plt.pause(0.001)
                except Exception:
                    pass
            if bool(args.debug_dump_lr) and int(args.debug_dump_every) > 0 and (step % int(args.debug_dump_every) == 0):
                _debug_print_lr(fr_body_raw, "raw")
                _debug_print_lr(fr_body_xform, "xform")
                _debug_print_lr(fr_gmr, "gmr")
            fps_mon.tick()

            if int(args.print_every) > 0 and (step % int(args.print_every) == 0):
                print(f"[teleop] frameIndex={frame_id} step={step} dt={dt*1000:.1f}ms t_action_ms={_now_ms()}")

            step += 1
            if rate is not None:
                rate.sleep()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，准备退出...")
    finally:
        kb.stop()
        _cleanup_wuji_hand_sim()
        if vmc_receiver is not None:
            try:
                vmc_receiver.close()
            except Exception:
                pass
        try:
            if video_writer is not None:
                video_writer.release()
                if video_filename:
                    print(f"[Video] saved: {video_filename}")
        except Exception:
            pass
        try:
            if viewer_ctx is not None:
                viewer_ctx.__exit__(None, None, None)
        except Exception:
            pass
        try:
            time.sleep(0.05)
            # Best-effort close both sources
            if use_xdmocap_body:
                udp_remove(int(mocap_index_body), str(dst_ip_body), int(dst_port_body))
            if int(mocap_index_hand) != int(mocap_index_body) or split_sources or (not use_xdmocap_body):
                try:
                    udp_remove(int(mocap_index_hand), str(dst_ip_hand), int(dst_port_hand))
                except Exception:
                    pass
            time.sleep(0.05)
            if use_xdmocap_body:
                udp_close(int(mocap_index_body))
            if int(mocap_index_hand) != int(mocap_index_body) or (not use_xdmocap_body):
                try:
                    udp_close(int(mocap_index_hand))
                except Exception:
                    pass
        except Exception:
            pass

    print("✅ teleop finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


