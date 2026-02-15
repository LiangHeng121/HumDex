#!/usr/bin/env python3
"""
离线可视化：data_record_xdmocap_oneclick.sh 录制的右手 hand_tracking_right (26D)。

用法示例：
  - 指定 task_dir（自动找最新 episode）：
      python deploy_real/viz_recorded_hand_right.py /path/to/deploy_real/twist2_demonstration/20260115_0905
  - 指定 episode_dir：
      python deploy_real/viz_recorded_hand_right.py /path/to/.../episode_0000
  - 指定 data.json：
      python deploy_real/viz_recorded_hand_right.py /path/to/.../episode_0000/data.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


HAND_JOINT_NAMES_26 = [
    "Wrist",
    "Palm",
    "ThumbMetacarpal",
    "ThumbProximal",
    "ThumbDistal",
    "ThumbTip",
    "IndexMetacarpal",
    "IndexProximal",
    "IndexIntermediate",
    "IndexDistal",
    "IndexTip",
    "MiddleMetacarpal",
    "MiddleProximal",
    "MiddleIntermediate",
    "MiddleDistal",
    "MiddleTip",
    "RingMetacarpal",
    "RingProximal",
    "RingIntermediate",
    "RingDistal",
    "RingTip",
    "LittleMetacarpal",
    "LittleProximal",
    "LittleIntermediate",
    "LittleDistal",
    "LittleTip",
]


def _connections_26() -> List[Tuple[int, int]]:
    # wrist -> palm
    conn: List[Tuple[int, int]] = [(0, 1)]

    # palm -> each finger base + chains
    def chain(base: int, ids: List[int]) -> None:
        # connect palm to base, and consecutive joints
        if ids:
            conn.append((base, ids[0]))
        for a, b in zip(ids[:-1], ids[1:]):
            conn.append((a, b))

    palm = 1
    chain(palm, [2, 3, 4, 5])  # thumb
    chain(palm, [6, 7, 8, 9, 10])  # index
    chain(palm, [11, 12, 13, 14, 15])  # middle
    chain(palm, [16, 17, 18, 19, 20])  # ring
    chain(palm, [21, 22, 23, 24, 25])  # little
    return conn


FINGERTIP_IDXS_26 = [5, 10, 15, 20, 25]
WRIST_IDX_26 = 0


def _resolve_data_json(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.name.endswith(".json"):
        return p
    if p.is_dir():
        # if task_dir, pick latest episode_*
        episodes = sorted([x for x in p.iterdir() if x.is_dir() and x.name.startswith("episode_")])
        if episodes:
            return (episodes[-1] / "data.json").resolve()
        # if episode_dir, use data.json inside
        candidate = p / "data.json"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"无法定位 data.json: {p}")


def _load_episode_items(data_json: Path) -> List[Dict[str, Any]]:
    with open(data_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    items = obj.get("data", None)
    if not isinstance(items, list):
        raise ValueError(f"data.json 格式不对：缺少 data(list)，path={data_json}")
    return items


def _extract_hand26_from_item(item: Dict[str, Any], *, side: str = "right") -> Optional[np.ndarray]:
    ht_key = "hand_tracking_right" if side.lower() == "right" else "hand_tracking_left"
    ht = item.get(ht_key, None)
    if not isinstance(ht, dict):
        return None
    if not bool(ht.get("is_active", False)):
        return None
    prefix = "RightHand" if side.lower() == "right" else "LeftHand"
    pts = np.zeros((26, 3), dtype=np.float32)
    for i, jn in enumerate(HAND_JOINT_NAMES_26):
        key = prefix + jn
        v = ht.get(key, None)
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            pts[i] = np.asarray(v[0], dtype=np.float32).reshape(3)
        else:
            pts[i] = np.nan
    if not np.isfinite(pts).any():
        return None
    return pts


def _set_axes_equal(ax, pts: np.ndarray) -> None:
    # keep xyz aspect ratio equal
    finite = np.isfinite(pts).all(axis=1)
    if not np.any(finite):
        return
    p = pts[finite]
    mins = p.min(axis=0)
    maxs = p.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.6 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> int:
    ap = argparse.ArgumentParser(description="离线可视化录制的 hand_tracking_right (26D)")
    ap.add_argument("path", type=str, help="task_dir / episode_dir / data.json 路径")
    ap.add_argument("--fps", type=float, default=30.0, help="播放帧率")
    ap.add_argument("--start", type=int, default=0, help="起始帧 index")
    ap.add_argument("--max_frames", type=int, default=-1, help="最多播放帧数（-1=全部）")
    ap.add_argument("--center_wrist", action="store_true", help="以 wrist 为原点居中（更稳定）")
    args = ap.parse_args()

    data_json = _resolve_data_json(args.path)
    items = _load_episode_items(data_json)
    conns = _connections_26()

    # matplotlib is optional; import lazily so script can fail with a clear message
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少 matplotlib，无法可视化。请先安装：pip install matplotlib") from e

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Recorded Hand (right) - {data_json.parent.name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # init artists
    pts0 = np.zeros((26, 3), dtype=np.float32)
    scat = ax.scatter(pts0[:, 0], pts0[:, 1], pts0[:, 2], s=20, c="k")
    scat_hi = ax.scatter([], [], [], s=50, c="r")  # wrist + fingertips
    lines = []
    for _ in conns:
        (ln,) = ax.plot([0, 0], [0, 0], [0, 0], c="k", lw=2)
        lines.append(ln)

    plt.ion()
    plt.show()

    start = max(0, int(args.start))
    end = len(items) if int(args.max_frames) <= 0 else min(len(items), start + int(args.max_frames))
    dt = 1.0 / max(1e-3, float(args.fps))

    for i in range(start, end):
        item = items[i]
        pts = _extract_hand26_from_item(item, side="right")
        if pts is None:
            # still advance, but keep previous drawing
            plt.pause(0.001)
            continue

        pts = pts.copy()
        if bool(args.center_wrist) and np.isfinite(pts[WRIST_IDX_26]).all():
            pts = pts - pts[WRIST_IDX_26].reshape(1, 3)

        # update scatter
        finite = np.isfinite(pts).all(axis=1)
        p = pts.copy()
        p[~finite] = 0.0
        scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])

        hi_ids = [WRIST_IDX_26] + FINGERTIP_IDXS_26
        hi = p[np.asarray(hi_ids, dtype=np.int32)]
        scat_hi._offsets3d = (hi[:, 0], hi[:, 1], hi[:, 2])

        # update lines
        for ln, (a, b) in zip(lines, conns):
            pa, pb = p[a], p[b]
            ln.set_data([pa[0], pb[0]], [pa[1], pb[1]])
            ln.set_3d_properties([pa[2], pb[2]])

        _set_axes_equal(ax, p)
        fig.canvas.draw_idle()
        plt.pause(0.001)
        time.sleep(dt)

    print(f"[done] played frames: {start}..{end-1}, data_json={data_json}")
    plt.ioff()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


