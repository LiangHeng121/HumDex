#!/usr/bin/env python3
"""
Standalone viewer: visualize hand_tracking_* from Redis in 3D (matplotlib).

Supports:
- mode=26 (recommended for "原始接收的 26D tracking 数据"):
  26 joints: Wrist, Palm, Thumb(4), Index(5), Middle(5), Ring(5), Little(5)
- mode=21:
  Convert the 26D dict -> 21D MediaPipe keypoints and draw.

Reads:
- hand_tracking_{left/right}_{robot_key}
Input format:
- JSON dict like {"is_active": True, "timestamp": 123, "LeftHandWrist": [[x,y,z],[qw,qx,qy,qz]], ...}

Colors:
- Wrist + fingertips in red; others in green.

This script is independent from Wuji hardware / open3d to avoid affecting control loops.
"""

import argparse
import json
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import redis


# 26D hand joint names (same as deploy_real/server_wuji_hand_redis.py)
HAND_JOINT_NAMES_26 = [
    "Wrist", "Palm",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip",
]

# 26 -> 21 mapping (MediaPipe order: Wrist + 5 fingers x 4)
MEDIAPIPE_MAPPING_26_TO_21 = [
    1,   # 0: Wrist (NOTE: historical mapping in this repo uses Palm index=1 as Wrist)
    2, 3, 4, 5,        # thumb
    6, 7, 8, 10,       # index (skip distal)
    11, 12, 13, 15,    # middle (skip distal)
    16, 17, 18, 20,    # ring (skip distal)
    21, 22, 23, 25,    # pinky (skip distal)
]

# MediaPipe 21 skeleton (same as server_wuji_hand_redis.py)
MP_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

TRACKING26_CONNECTIONS = [
    # Wrist -> Palm (some pipelines use same point, but keep for completeness)
    (0, 1),
    # Thumb: Wrist -> ThumbMetacarpal -> ThumbProximal -> ThumbDistal -> ThumbTip
    (0, 2), (2, 3), (3, 4), (4, 5),
    # Index: Wrist -> IndexMetacarpal -> IndexProximal -> IndexIntermediate -> IndexDistal -> IndexTip
    (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    # Middle
    (0, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    # Ring
    (0, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    # Little
    (0, 21), (21, 22), (22, 23), (23, 24), (24, 25),
]


def hand_26d_to_mediapipe_21d(hand_data_dict: Dict[str, Any], hand_side: str = "left") -> np.ndarray:
    """
    Convert 26D dict -> 21x3 mediapipe points.
    Output is wrist-local (wrist at origin), consistent with retarget pipeline usage.
    """
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"

    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        v = hand_data_dict.get(key, None)
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            joint_positions_26[i] = np.asarray(v[0], dtype=np.float32).reshape(3)
        else:
            joint_positions_26[i] = 0.0

    pts21 = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21].copy()
    wrist = pts21[0].copy()
    pts21 = pts21 - wrist
    return pts21


def hand_tracking26_points(hand_data_dict: Dict[str, Any], hand_side: str = "left") -> np.ndarray:
    """
    Extract raw 26 joint points from hand_tracking dict (26,3).
    """
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"
    pts26 = np.zeros((26, 3), dtype=np.float32)
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        v = hand_data_dict.get(key, None)
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            pts26[i] = np.asarray(v[0], dtype=np.float32).reshape(3)
        else:
            pts26[i] = 0.0
    return pts26


def _parse_hand_tracking_payload(raw: bytes) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Returns (is_active, dict_or_none)
    """
    try:
        s = raw.decode("utf-8")
        d = json.loads(s)
        if not isinstance(d, dict):
            return False, None
        is_active = bool(d.get("is_active", False))
        return is_active, d
    except Exception:
        return False, None


def main() -> int:
    ap = argparse.ArgumentParser(description="3D viewer for Redis hand_tracking_* (26D -> 21D)")
    ap.add_argument("--redis_ip", type=str, default="localhost")
    ap.add_argument("--redis_port", type=int, default=6379)
    ap.add_argument("--robot_key", type=str, default="unitree_g1_with_hands")
    ap.add_argument("--hand_side", type=str, default="right", choices=["left", "right"])
    ap.add_argument("--mode", choices=["26", "21", "bvh"], default="26",
                    help="26=visualize raw tracking26 joints; 21=convert to MediaPipe 21 keypoints; bvh=visualize BVH-style joints (LeftHand/LeftIndexFinger...) (default: 26)")
    ap.add_argument("--fps", type=float, default=20.0, help="viewer refresh rate (Hz)")
    ap.add_argument("--lines", action="store_true", help="draw skeleton lines")
    ap.add_argument("--labels", action="store_true", help="draw point indices (mode=26 => 0-25, mode=21 => 0-20)")
    ap.add_argument("--coord", choices=["wrist_local", "world_like"], default="wrist_local",
                    help="wrist_local: wrist at origin (only applies to mode=21); world_like: keep absolute")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply coordinates for display")
    ap.add_argument("--z_up", action="store_true", help="force z-up view (matplotlib default is z-up anyway)")
    args = ap.parse_args()

    if args.mode == "bvh":
        key = f"hand_bvh_{args.hand_side}_{args.robot_key}"
    else:
        key = f"hand_tracking_{args.hand_side}_{args.robot_key}"
    r = redis.Redis(host=args.redis_ip, port=int(args.redis_port), decode_responses=False)

    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print(f"❌ 缺少 matplotlib，无法可视化：{e}")
        print("   解决：pip install matplotlib")
        return 2

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title(f"hand{args.mode}d_redis_{args.hand_side}")

    # artists
    scat_red = None
    scat_green = None
    line_artists = []
    text_artists = []

    def clear_artists():
        nonlocal scat_red, scat_green, line_artists, text_artists
        if scat_red is not None:
            scat_red.remove()
            scat_red = None
        if scat_green is not None:
            scat_green.remove()
            scat_green = None
        for ln in line_artists:
            try:
                ln.remove()
            except Exception:
                pass
        for t in text_artists:
            try:
                t.remove()
            except Exception:
                pass
        line_artists = []
        text_artists = []

    def set_equal_aspect(P: np.ndarray):
        mn = np.min(P, axis=0)
        mx = np.max(P, axis=0)
        mid = 0.5 * (mn + mx)
        r = 0.55 * float(np.max(mx - mn) + 1e-9)
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)

    tip_idxs_21 = {0, 4, 8, 12, 16, 20}
    tip_idxs_26 = {0, 5, 10, 15, 20, 25}
    # BVH-style: 20 joints (Hand + 4*4 + Thumb*3)
    tip_idxs_bvh = {0, 3, 7, 11, 15, 19}
    BVH_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3),              # thumb: Hand -> ThumbFinger -> ThumbFinger1 -> ThumbFinger2
        (0, 4), (4, 5), (5, 6), (6, 7),      # index
        (0, 8), (8, 9), (9, 10), (10, 11),   # middle
        (0, 12), (12, 13), (13, 14), (14, 15),  # ring
        (0, 16), (16, 17), (17, 18), (18, 19),  # pinky
    ]
    last_print = 0.0
    dt = 1.0 / max(1e-6, float(args.fps))

    print(f"👁️  Viewer started. mode={args.mode} Redis={args.redis_ip}:{args.redis_port} key={key!r} refresh={args.fps:.1f}Hz")
    print("   鼠标拖拽旋转；滚轮缩放；关闭窗口即可退出。")

    try:
        while plt.fignum_exists(fig.number):
            raw = r.get(key)
            if raw is None:
                if (time.time() - last_print) > 1.0:
                    print(f"⏳ 等待 Redis key: {key}")
                    last_print = time.time()
                plt.pause(dt)
                continue

            is_active, d = _parse_hand_tracking_payload(raw)
            if not is_active or d is None:
                plt.pause(dt)
                continue

            if args.mode == "bvh":
                # Expect keys like LeftHand/LeftIndexFinger... in the payload, each value is [[x,y,z],[qw,qx,qy,qz]]
                pfx = "Left" if args.hand_side == "left" else "Right"
                bvh_names = [
                    f"{pfx}Hand",
                    f"{pfx}ThumbFinger", f"{pfx}ThumbFinger1", f"{pfx}ThumbFinger2",
                    f"{pfx}IndexFinger", f"{pfx}IndexFinger1", f"{pfx}IndexFinger2", f"{pfx}IndexFinger3",
                    f"{pfx}MiddleFinger", f"{pfx}MiddleFinger1", f"{pfx}MiddleFinger2", f"{pfx}MiddleFinger3",
                    f"{pfx}RingFinger", f"{pfx}RingFinger1", f"{pfx}RingFinger2", f"{pfx}RingFinger3",
                    f"{pfx}PinkyFinger", f"{pfx}PinkyFinger1", f"{pfx}PinkyFinger2", f"{pfx}PinkyFinger3",
                ]
                pts = []
                for n in bvh_names:
                    vv = d.get(n, None)
                    if isinstance(vv, (list, tuple)) and len(vv) >= 1:
                        pts.append(np.asarray(vv[0], dtype=np.float32).reshape(3))
                    else:
                        pts.append(np.zeros(3, dtype=np.float32))
                pts = np.stack(pts, axis=0)  # (20,3)
                P = (pts.astype(np.float32) * float(args.scale)).reshape(20, 3)
                tip_idxs = tip_idxs_bvh
                conns = BVH_CONNECTIONS
            elif args.mode == "21":
                pts = hand_26d_to_mediapipe_21d(d, hand_side=args.hand_side)  # (21,3)
                if args.coord == "world_like":
                    # best-effort: add back the original (mapped) wrist (repo mapping uses Palm index=1)
                    hand_side_prefix = "LeftHand" if args.hand_side == "left" else "RightHand"
                    v = d.get(hand_side_prefix + "Palm", None)
                    if isinstance(v, (list, tuple)) and len(v) >= 1:
                        wrist_abs = np.asarray(v[0], dtype=np.float32).reshape(3)
                        pts = pts + wrist_abs
                P = (pts.astype(np.float32) * float(args.scale)).reshape(21, 3)
                tip_idxs = tip_idxs_21
                conns = MP_HAND_CONNECTIONS
            else:
                pts = hand_tracking26_points(d, hand_side=args.hand_side)  # (26,3)
                # coord option does not apply for mode=26; it's already "raw"
                P = (pts.astype(np.float32) * float(args.scale)).reshape(26, 3)
                tip_idxs = tip_idxs_26
                conns = TRACKING26_CONNECTIONS

            # redraw
            clear_artists()
            ax.cla()
            ax.set_title(f"hand{args.mode}d ({args.hand_side})  key={key}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            red = P[list(sorted(tip_idxs))]
            green = np.array([P[i] for i in range(P.shape[0]) if i not in tip_idxs], dtype=np.float32)
            scat_red = ax.scatter(red[:, 0], red[:, 1], red[:, 2], s=50, c="r", depthshade=True)
            scat_green = ax.scatter(green[:, 0], green[:, 1], green[:, 2], s=35, c="g", depthshade=True)

            if args.lines:
                for a, b in conns:
                    pa, pb = P[a], P[b]
                    (ln,) = ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], c="0.6", linewidth=2.0)
                    line_artists.append(ln)

            if args.labels:
                for i in range(P.shape[0]):
                    p = P[i]
                    txt = ax.text(p[0], p[1], p[2], str(i), fontsize=8, color="k")
                    text_artists.append(txt)

            set_equal_aspect(P)
            plt.pause(dt)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            plt.ioff()
            plt.close(fig)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


