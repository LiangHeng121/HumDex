#!/usr/bin/env python3
"""
Estimate a global similarity transform between a pose CSV and a BVH that represent
the SAME motion segment.

We fit:
    p_bvh ~= s * (R @ p_csv) + t

BVH positions are loaded via our GMR BVH loader (same convention used in replay scripts):
  - FK to global
  - apply fixed BVH->GMR axis rotation
  - cm -> m

CSV positions are loaded via pose_csv_loader.load_pose_csv_frames (supports both formats):
  - body_/lhand_/rhand_ + _px/_qw
  - motionData style: "Hips position X(m)" + "Hips quaternion W"
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np


CORE_JOINTS = [
    "Hips",
    "Spine2",
    "Head",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFoot",
    "RightFoot",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_repo_on_path() -> None:
    rr = _repo_root()
    if rr not in sys.path:
        sys.path.insert(0, rr)


def _load_csv_frames(csv_file: str, fmt: str) -> Tuple[List[Dict[str, Any]], float]:
    _ensure_repo_on_path()
    from deploy_real.pose_csv_loader import load_pose_csv_frames, gmr_rename_and_footmod  # type: ignore

    frames_raw, meta = load_pose_csv_frames(csv_file, include_body=True, include_lhand=False, include_rhand=False)
    frames = [gmr_rename_and_footmod(fr, fmt=fmt) for fr in frames_raw]
    fps = float(meta.fps) if meta.fps is not None else float("nan")
    return frames, fps


def _load_bvh_frames(bvh_file: str, fmt: str) -> List[Dict[str, Any]]:
    _ensure_repo_on_path()
    from deploy_real.replay_bvh_body_to_redis import _load_bvh_frames_via_gmr  # type: ignore

    frames, _ = _load_bvh_frames_via_gmr(bvh_file, fmt=fmt)
    return frames


def _stack_correspondences(
    csv_frames: List[Dict[str, Any]],
    bvh_frames: List[Dict[str, Any]],
    csv_start: int,
    bvh_start: int,
    n_frames: int,
    stride: int,
    joints: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    used_joints: List[str] = []
    used_pairs = 0

    max_n = min(len(csv_frames) - csv_start, len(bvh_frames) - bvh_start)
    n_use = max_n if n_frames < 0 else min(max_n, int(n_frames))
    if n_use <= 0:
        raise ValueError(f"no overlapping frames: csv_start={csv_start}, bvh_start={bvh_start}, max_n={max_n}")

    for k in joints:
        used_joints.append(k)

    for i in range(0, n_use, max(1, int(stride))):
        fc = csv_frames[csv_start + i]
        fb = bvh_frames[bvh_start + i]
        for j in joints:
            if j not in fc or j not in fb:
                continue
            Xs.append(np.asarray(fc[j][0], dtype=np.float32).reshape(3))
            Ys.append(np.asarray(fb[j][0], dtype=np.float32).reshape(3))
            used_pairs += 1

    if used_pairs < 3:
        raise ValueError(f"too few correspondences: {used_pairs} (<3). Try different joint set or offsets.")
    return np.stack(Xs, axis=0), np.stack(Ys, axis=0), used_joints, used_pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate similarity transform between CSV and BVH (same motion)")
    ap.add_argument("--csv_file", type=str, required=True)
    ap.add_argument("--bvh_file", type=str, required=True)
    ap.add_argument("--format", choices=["lafan1", "nokov"], default="nokov")
    ap.add_argument("--csv_start", type=int, default=0)
    ap.add_argument("--bvh_start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=240, help="How many frames to use (-1 for all overlap)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--no_scale", action="store_true", help="Fit rigid transform only (s=1)")
    ap.add_argument("--use_all_common_joints", action="store_true", help="Use all joints in intersection instead of CORE_JOINTS")
    args = ap.parse_args()

    csv_path = args.csv_file
    bvh_path = args.bvh_file
    if not os.path.isabs(csv_path):
        cand = os.path.join(_repo_root(), csv_path)
        if os.path.exists(cand):
            csv_path = cand
    if not os.path.isabs(bvh_path):
        cand = os.path.join(_repo_root(), bvh_path)
        if os.path.exists(cand):
            bvh_path = cand
    if not os.path.exists(csv_path):
        print(f"❌ 找不到 CSV: {args.csv_file}", file=sys.stderr)
        return 2
    if not os.path.exists(bvh_path):
        print(f"❌ 找不到 BVH: {args.bvh_file}", file=sys.stderr)
        return 2

    csv_frames, csv_fps = _load_csv_frames(csv_path, fmt=args.format)
    bvh_frames = _load_bvh_frames(bvh_path, fmt=args.format)
    print("======================================================================")
    print("Estimate CSV -> BVH similarity transform")
    print("======================================================================")
    print(f"csv_file: {csv_path}")
    print(f"bvh_file: {bvh_path}")
    print(f"format  : {args.format}")
    print(f"csv_fps : {csv_fps}")
    print(f"csv_len : {len(csv_frames)}")
    print(f"bvh_len : {len(bvh_frames)}")

    if args.use_all_common_joints:
        common = sorted(list(set(csv_frames[0].keys()).intersection(set(bvh_frames[0].keys()))))
        # remove auxiliary joints that may bias
        common = [j for j in common if not j.endswith("FootMod")]
        joints = common
    else:
        joints = list(CORE_JOINTS)

    _ensure_repo_on_path()
    from deploy_real.pose_csv_loader import umeyama_similarity_transform  # type: ignore

    X, Y, used_joints, used_pairs = _stack_correspondences(
        csv_frames,
        bvh_frames,
        csv_start=int(args.csv_start),
        bvh_start=int(args.bvh_start),
        n_frames=int(args.frames),
        stride=int(args.stride),
        joints=joints,
    )

    s, Rm, t = umeyama_similarity_transform(X, Y, with_scale=(not bool(args.no_scale)))
    Yhat = (float(s) * (X @ Rm.T)) + t.reshape(1, 3)
    resid = Y - Yhat
    rms = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))

    print("------------------------------------------------------------------")
    print("Fit result:  p_bvh ~= s*(R@p_csv) + t")
    print(f"pairs_used: {used_pairs}")
    print(f"joints    : {used_joints}")
    print(f"s         : {s:.8f}")
    print(f"det(R)    : {float(np.linalg.det(Rm)):.6f}")
    print(f"t         : {t.tolist()}")
    print("R =")
    print(Rm)
    print(f"RMS error : {rms:.6f} m")
    print("------------------------------------------------------------------")
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = float(s) * Rm
    T[:3, 3] = t
    print("4x4 (includes scale in rotation block):")
    print(T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


