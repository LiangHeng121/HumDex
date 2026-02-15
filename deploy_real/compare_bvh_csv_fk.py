#!/usr/bin/env python3
"""
Compare BVH FK'ed global joint positions with pose CSV positions.

Goal: answer "BVH FK 后的数据和 CSV 数据应该一样吗？"
We provide:
- Optional similarity fit: p_bvh ~= s*(R@p_csv) + t
- Optional time offset search between BVH and CSV
- Overall RMS / max error, plus per-joint stats

Notes:
- BVH is loaded via deploy_real/replay_bvh_body_to_redis._load_bvh_frames_via_gmr
  (includes FK, axis rotation, and cm->m conversion)
- CSV is loaded via deploy_real/pose_csv_loader.load_pose_csv_frames + gmr_rename_and_footmod
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


DEFAULT_CORE = [
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


def _resolve_path(p: str) -> str:
    p = str(p)
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(_repo_root(), p)
    if os.path.exists(cand):
        return cand
    return p


def _load_bvh_frames(bvh_file: str, fmt: str) -> List[Dict[str, Any]]:
    _ensure_repo_on_path()
    from deploy_real.replay_bvh_body_to_redis import _load_bvh_frames_via_gmr  # type: ignore

    frames, _ = _load_bvh_frames_via_gmr(bvh_file, fmt=fmt)
    return frames


def _load_csv_frames(csv_file: str, fmt: str) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    _ensure_repo_on_path()
    from deploy_real.pose_csv_loader import load_pose_csv_frames, gmr_rename_and_footmod  # type: ignore

    frames_raw, meta = load_pose_csv_frames(csv_file, include_body=True, include_lhand=False, include_rhand=False)
    frames = [gmr_rename_and_footmod(fr, fmt=fmt) for fr in frames_raw]
    fps = float(meta.fps) if meta.fps is not None else None
    return frames, fps


def _pos(fr: Dict[str, Any], j: str) -> Optional[np.ndarray]:
    if j not in fr:
        return None
    v = fr[j]
    if not isinstance(v, (list, tuple)) or len(v) < 1:
        return None
    return np.asarray(v[0], dtype=np.float32).reshape(3)


def _common_joints(fc0: Dict[str, Any], fb0: Dict[str, Any]) -> List[str]:
    js = sorted(list(set(fc0.keys()).intersection(set(fb0.keys()))))
    return [j for j in js if not str(j).endswith("FootMod")]


def _stack_pairs(
    csv_frames: List[Dict[str, Any]],
    bvh_frames: List[Dict[str, Any]],
    joints: List[str],
    csv_start: int,
    bvh_start: int,
    n_frames: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    max_n = min(len(csv_frames) - csv_start, len(bvh_frames) - bvh_start)
    n_use = max_n if n_frames < 0 else min(max_n, int(n_frames))
    for i in range(0, n_use, max(1, int(stride))):
        fc = csv_frames[csv_start + i]
        fb = bvh_frames[bvh_start + i]
        for j in joints:
            pc = _pos(fc, j)
            pb = _pos(fb, j)
            if pc is None or pb is None:
                continue
            xs.append(pc)
            ys.append(pb)
    if len(xs) < 3:
        raise ValueError(f"need >=3 correspondences, got {len(xs)}")
    return np.stack(xs, axis=0), np.stack(ys, axis=0), n_use


def _apply_similarity_to_pos(P: np.ndarray, s: float, Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    # P: (N,3), Rm: (3,3), t: (3,)
    return (float(s) * (P @ Rm.T)) + t.reshape(1, 3)


def _error_stats(
    csv_frames: List[Dict[str, Any]],
    bvh_frames: List[Dict[str, Any]],
    joints: List[str],
    s: float,
    Rm: np.ndarray,
    t: np.ndarray,
    csv_start: int,
    bvh_start: int,
    n_frames: int,
    stride: int,
) -> Tuple[float, float, Dict[str, Tuple[float, float, int]]]:
    # returns: overall_rms, overall_max, per_joint (rms,max,count)
    errs_all: List[float] = []
    per: Dict[str, List[float]] = {j: [] for j in joints}
    max_n = min(len(csv_frames) - csv_start, len(bvh_frames) - bvh_start)
    n_use = max_n if n_frames < 0 else min(max_n, int(n_frames))
    for i in range(0, n_use, max(1, int(stride))):
        fc = csv_frames[csv_start + i]
        fb = bvh_frames[bvh_start + i]
        for j in joints:
            pc = _pos(fc, j)
            pb = _pos(fb, j)
            if pc is None or pb is None:
                continue
            pc2 = (float(s) * (Rm @ pc)) + t
            e = float(np.linalg.norm(pb - pc2))
            errs_all.append(e)
            per[j].append(e)
    if not errs_all:
        raise ValueError("no comparable points found")
    arr = np.asarray(errs_all, dtype=np.float64)
    rms = float(np.sqrt(np.mean(arr * arr)))
    mx = float(np.max(arr))
    per_out: Dict[str, Tuple[float, float, int]] = {}
    for j, es in per.items():
        if not es:
            continue
        a = np.asarray(es, dtype=np.float64)
        per_out[j] = (float(np.sqrt(np.mean(a * a))), float(np.max(a)), int(a.shape[0]))
    return rms, mx, per_out


def _search_time_offset(
    csv_frames: List[Dict[str, Any]],
    bvh_frames: List[Dict[str, Any]],
    joints: List[str],
    s: float,
    Rm: np.ndarray,
    t: np.ndarray,
    csv_start: int,
    bvh_start: int,
    n_frames: int,
    stride: int,
    max_offset: int,
) -> Tuple[int, float]:
    """
    Search integer offset k such that comparing:
      csv[i+csv_start] vs bvh[i+bvh_start+k]
    yields minimal RMS (positions only).
    """
    best_k = 0
    best_rms = float("inf")
    for k in range(-int(max_offset), int(max_offset) + 1):
        cs = csv_start
        bs = bvh_start + k
        if bs < 0:
            continue
        try:
            rms, _, _ = _error_stats(
                csv_frames,
                bvh_frames,
                joints,
                s=s,
                Rm=Rm,
                t=t,
                csv_start=cs,
                bvh_start=bs,
                n_frames=n_frames,
                stride=stride,
            )
        except Exception:
            continue
        if rms < best_rms:
            best_rms = rms
            best_k = k
    return int(best_k), float(best_rms)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare BVH FK joint positions vs CSV joint positions")
    ap.add_argument("--csv_file", type=str, required=True)
    ap.add_argument("--bvh_file", type=str, required=True)
    ap.add_argument("--format", choices=["lafan1", "nokov"], default="nokov")
    ap.add_argument("--csv_start", type=int, default=0)
    ap.add_argument("--bvh_start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=240, help="How many frames to compare (-1 for all overlap)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--use_all_common_joints", action="store_true")
    ap.add_argument("--no_fit", action="store_true", help="Do not fit s,R,t (use identity)")
    ap.add_argument("--no_scale", action="store_true", help="Fit rigid only (s=1)")
    ap.add_argument("--search_time_offset", type=int, default=0, help="Search best BVH offset in [-k,k] after fitting")
    args = ap.parse_args()

    csv_path = _resolve_path(args.csv_file)
    bvh_path = _resolve_path(args.bvh_file)
    if not os.path.exists(csv_path):
        print(f"❌ 找不到 CSV: {args.csv_file}")
        return 2
    if not os.path.exists(bvh_path):
        print(f"❌ 找不到 BVH: {args.bvh_file}")
        return 2

    csv_frames, csv_fps = _load_csv_frames(csv_path, fmt=args.format)
    bvh_frames = _load_bvh_frames(bvh_path, fmt=args.format)
    print("======================================================================")
    print("Compare BVH(FK) vs CSV positions")
    print("======================================================================")
    print(f"csv_file: {csv_path}")
    print(f"bvh_file: {bvh_path}")
    print(f"format  : {args.format}")
    print(f"csv_fps : {csv_fps}")
    print(f"csv_len : {len(csv_frames)}")
    print(f"bvh_len : {len(bvh_frames)}")

    if args.use_all_common_joints:
        joints = _common_joints(csv_frames[0], bvh_frames[0])
    else:
        joints = list(DEFAULT_CORE)

    # fit similarity on chosen window
    s = 1.0
    Rm = np.eye(3, dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)
    if not bool(args.no_fit):
        _ensure_repo_on_path()
        from deploy_real.pose_csv_loader import umeyama_similarity_transform  # type: ignore

        X, Y, n_use = _stack_pairs(
            csv_frames,
            bvh_frames,
            joints=joints,
            csv_start=int(args.csv_start),
            bvh_start=int(args.bvh_start),
            n_frames=int(args.frames),
            stride=int(args.stride),
        )
        s, Rm, t = umeyama_similarity_transform(X, Y, with_scale=(not bool(args.no_scale)))
        print("[fit] p_bvh ~= s*(R@p_csv)+t")
        print(f"  s={float(s):.8f}, det(R)={float(np.linalg.det(Rm)):.3f}, t={t.tolist()}")
        print(f"  R=\\n{Rm}")

    best_offset = 0
    if int(args.search_time_offset) > 0:
        best_offset, best_rms = _search_time_offset(
            csv_frames,
            bvh_frames,
            joints=joints,
            s=s,
            Rm=Rm,
            t=t,
            csv_start=int(args.csv_start),
            bvh_start=int(args.bvh_start),
            n_frames=int(args.frames),
            stride=int(args.stride),
            max_offset=int(args.search_time_offset),
        )
        print(f"[time_offset_search] best_bvh_offset={best_offset} (bvh_start += {best_offset}), rms={best_rms:.6f} m")

    rms, mx, per = _error_stats(
        csv_frames,
        bvh_frames,
        joints=joints,
        s=s,
        Rm=Rm,
        t=t,
        csv_start=int(args.csv_start),
        bvh_start=int(args.bvh_start) + int(best_offset),
        n_frames=int(args.frames),
        stride=int(args.stride),
    )
    print("------------------------------------------------------------------")
    print(f"[compare] overall_rms={rms:.6f} m, overall_max={mx:.6f} m, joints_used={len(per)}/{len(joints)}")
    # show worst joints
    worst = sorted(per.items(), key=lambda kv: kv[1][0], reverse=True)[:10]
    print("[compare] worst joints (rms, max, count):")
    for j, (jr, jm, cnt) in worst:
        print(f"  {j:>14s}: rms={jr:.6f} m, max={jm:.6f} m, n={cnt}")
    print("------------------------------------------------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


