#!/usr/bin/env python3
"""
Derive a one-time calibration from a paired (CSV, BVH) recording of the same motion,
so later replays can run with CSV only (no BVH required).

It writes a calib.json containing:
- a fixed position mapping (default: (x,y,z)->(-x,z,y) ) as pos.matrix
- optional BVH-like rotation into GMR world (pos.apply_bvh_like_rotation=true)
- a per-joint quaternion fix map q_fix_j (default: right-multiply), learned from BVH(FK) quats

Typical usage (for your dataset):
python deploy_real/derive_csv_calib_from_bvh.py \
  --csv_file xdmocap/data/motionData_20260108210128.csv \
  --bvh_file xdmocap/data/motionData_20260108210128.bvh \
  --format nokov \
  --out_json xdmocap/data/csv_calib_motionData_20260108210128.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ang_deg_between(q_a: np.ndarray, q_b: np.ndarray, *, quat_mul_wxyz, quat_conj_wxyz, quat_normalize_wxyz) -> float:
    qa = quat_normalize_wxyz(np.asarray(q_a, dtype=np.float32).reshape(4))
    qb = quat_normalize_wxyz(np.asarray(q_b, dtype=np.float32).reshape(4))
    qd = quat_mul_wxyz(qa, quat_conj_wxyz(qb))
    qd = quat_normalize_wxyz(qd)
    w = float(np.clip(qd[0], -1.0, 1.0))
    return float(2.0 * np.arccos(abs(w)) * 180.0 / np.pi)


def _markley_mean(rels: List[np.ndarray]) -> np.ndarray:
    A = np.zeros((4, 4), dtype=np.float64)
    for q in rels:
        qq = np.asarray(q, dtype=np.float64).reshape(4)
        A += np.outer(qq, qq)
    A /= float(len(rels))
    w, V = np.linalg.eigh(A)
    q = V[:, int(np.argmax(w))].astype(np.float32)
    # standardize sign
    if q[0] < 0:
        q = -q
    return q


def main() -> int:
    p = argparse.ArgumentParser(description="Derive a CSV->GMR calibration json from paired CSV+BVH (same motion)")
    p.add_argument("--csv_file", type=str, required=True)
    p.add_argument("--bvh_file", type=str, required=True)
    p.add_argument("--format", choices=["lafan1", "nokov"], default="nokov")
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--pos_mode", choices=["fixed_negx_zy", "fit_similarity"], default="fixed_negx_zy")
    p.add_argument("--search_time_offset", type=int, default=20, help="search best BVH offset in [-k,k] (default: 20)")
    p.add_argument("--align_frames", type=int, default=240)
    p.add_argument("--align_stride", type=int, default=2)
    p.add_argument("--no_bvh_like_rotation", action="store_true", help="Do NOT apply BVH->GMR fixed rotation in the calib (NOT recommended)")
    args = p.parse_args()

    repo = _repo_root()
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from deploy_real.pose_csv_loader import (
        CSV_POS_NEGX_Z_Y_M,
        load_pose_csv_frames,
        gmr_rename_and_footmod,
        apply_pos_matrix,
        apply_bvh_like_coordinate_transform,
        umeyama_similarity_transform,
        apply_similarity_transform_frame,
        quat_normalize_wxyz,
        quat_mul_wxyz,
        quat_conj_wxyz,
        apply_quat_right_multiply_per_joint,
        apply_quat_left_multiply_per_joint,
    )
    from deploy_real.replay_bvh_body_to_redis import _load_bvh_frames_fk_raw_world

    csv_path = args.csv_file
    if not os.path.isabs(csv_path):
        cand = os.path.join(repo, csv_path)
        if os.path.exists(cand):
            csv_path = cand
    bvh_path = args.bvh_file
    if not os.path.isabs(bvh_path):
        cand = os.path.join(repo, bvh_path)
        if os.path.exists(cand):
            bvh_path = cand
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    if not os.path.exists(bvh_path):
        raise FileNotFoundError(bvh_path)

    csv_raw, _meta = load_pose_csv_frames(csv_path, include_body=True, include_lhand=False, include_rhand=False, max_frames=-1)
    csv_frames = [gmr_rename_and_footmod(fr, fmt=args.format) for fr in csv_raw]
    bvh_raw, _hh = _load_bvh_frames_fk_raw_world(bvh_path, fmt=args.format)

    # choose joint set for positional pairing
    core = [
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

    # Step A: position alignment into BVH raw world
    best_off = 0
    pos_s = 1.0
    pos_R = np.eye(3, dtype=np.float32)
    pos_t = np.zeros(3, dtype=np.float32)

    if args.pos_mode == "fixed_negx_zy":
        # Use the fixed rotation as a proper world rotation: apply to BOTH positions and quats.
        # This matches the BVH-alignment path where similarity R also rotates quats.
        R_fix = CSV_POS_NEGX_Z_Y_M.astype(np.float32)
        t_fix = np.zeros(3, dtype=np.float32)
        s_fix = 1.0
        csv_pos_aligned = [apply_similarity_transform_frame(fr, s_fix, R_fix, t_fix) for fr in csv_frames]
        # search best offset by RMS (positions only)
        k = max(0, int(args.search_time_offset))
        stride = max(1, int(args.align_stride))
        n_req = int(args.align_frames)
        best = None
        for off in range(-k, k + 1):
            b0 = off
            if b0 < 0:
                continue
            max_n = min(len(csv_pos_aligned), len(bvh_raw) - b0)
            n_use = max_n if n_req < 0 else min(max_n, n_req)
            if n_use <= 0:
                continue
            xs = []
            ys = []
            for i in range(0, n_use, stride):
                fc = csv_pos_aligned[i]
                fb = bvh_raw[b0 + i]
                for j in core:
                    if j in fc and j in fb:
                        xs.append(np.asarray(fc[j][0], np.float32).reshape(3))
                        ys.append(np.asarray(fb[j][0], np.float32).reshape(3))
            if len(xs) < 3:
                continue
            X = np.stack(xs)
            Y = np.stack(ys)
            rms = float(np.sqrt(np.mean(np.sum((Y - X) ** 2, axis=1))))
            cand = (rms, off, n_use, len(xs))
            if best is None or cand[0] < best[0]:
                best = cand
        if best is None:
            raise RuntimeError("failed to search time offset")
        rms, best_off, n_use, pairs = best
        print(f"[calib] pos_mode=fixed_negx_zy: best_bvh_offset={best_off} rms={rms:.6f} m pairs={pairs} frames_used={n_use}")
        csv_aligned = csv_pos_aligned
        pos_s, pos_R, pos_t = float(s_fix), R_fix, t_fix
    else:
        # fit similarity transform (s,R,t) and search offset
        k = max(0, int(args.search_time_offset))
        stride = max(1, int(args.align_stride))
        n_req = int(args.align_frames)

        def _fit(off: int):
            b0 = off
            if b0 < 0:
                return None
            max_n = min(len(csv_frames), len(bvh_raw) - b0)
            n_use = max_n if n_req < 0 else min(max_n, n_req)
            if n_use <= 0:
                return None
            xs = []
            ys = []
            for i in range(0, n_use, stride):
                fc = csv_frames[i]
                fb = bvh_raw[b0 + i]
                for j in core:
                    if j in fc and j in fb:
                        xs.append(np.asarray(fc[j][0], np.float32).reshape(3))
                        ys.append(np.asarray(fb[j][0], np.float32).reshape(3))
            if len(xs) < 3:
                return None
            X = np.stack(xs)
            Y = np.stack(ys)
            s, Rm, t = umeyama_similarity_transform(X, Y, with_scale=True)
            Yhat = (float(s) * (X @ Rm.T)) + t.reshape(1, 3)
            rms = float(np.sqrt(np.mean(np.sum((Y - Yhat) ** 2, axis=1))))
            return rms, off, n_use, len(xs), float(s), Rm.astype(np.float32), t.astype(np.float32)

        best = None
        for off in range(-k, k + 1):
            r = _fit(off)
            if r is None:
                continue
            if best is None or r[0] < best[0]:
                best = r
        if best is None:
            raise RuntimeError("failed to fit similarity transform")
        rms, best_off, n_use, pairs, pos_s, pos_R, pos_t = best
        print(f"[calib] pos_mode=fit_similarity: best_bvh_offset={best_off} rms={rms:.6f} m pairs={pairs} frames_used={n_use}")
        print(f"  s={pos_s:.8f} t={pos_t.tolist()}\n  R=\n{pos_R}")
        csv_aligned = [apply_similarity_transform_frame(fr, pos_s, pos_R, pos_t) for fr in csv_frames]

    # Step B: convert BOTH to GMR world (optional but recommended)
    apply_bvh_like = (not bool(args.no_bvh_like_rotation))
    if apply_bvh_like:
        csv_gmr = [apply_bvh_like_coordinate_transform(fr, pos_unit="m", apply_rotation=True) for fr in csv_aligned]
        bvh_gmr = [apply_bvh_like_coordinate_transform(fr, pos_unit="m", apply_rotation=True) for fr in bvh_raw]
    else:
        csv_gmr = list(csv_aligned)
        bvh_gmr = list(bvh_raw)

    # Step C: estimate per-joint q_fix_j (right-multiply by default)
    side = "right"
    stride = max(1, int(args.align_stride))
    n_req = int(args.align_frames)
    max_n = min(len(csv_gmr), len(bvh_gmr) - best_off)
    n_use = max_n if n_req < 0 else min(max_n, n_req)
    if n_use <= 0:
        raise RuntimeError("no overlap frames to estimate quat fix")

    base_fb = bvh_gmr[best_off]
    joint_list = sorted([j for j in csv_gmr[0].keys() if j in base_fb and (not str(j).endswith("FootMod"))])
    rels_map: Dict[str, List[np.ndarray]] = {}
    qref_map: Dict[str, np.ndarray] = {}
    for i in range(0, n_use, stride):
        fb = bvh_gmr[best_off + i]
        fc = csv_gmr[i]
        for j in joint_list:
            if j not in fb or j not in fc:
                continue
            qb = quat_normalize_wxyz(np.asarray(fb[j][1], np.float32).reshape(4))
            qc = quat_normalize_wxyz(np.asarray(fc[j][1], np.float32).reshape(4))
            # right-multiply: qb ~= qc ⊗ q_fix => q_fix ~= inv(qc) ⊗ qb
            qrel = quat_mul_wxyz(quat_conj_wxyz(qc), qb)
            qrel = quat_normalize_wxyz(qrel)
            if j not in rels_map:
                rels_map[j] = []
                qref_map[j] = qrel.copy()
            else:
                if float(np.dot(qrel, qref_map[j])) < 0.0:
                    qrel = -qrel
            rels_map[j].append(qrel)

    qfix_map: Dict[str, np.ndarray] = {}
    for j, rels in rels_map.items():
        if len(rels) < 3:
            continue
        q = _markley_mean(rels)
        # align sign to reference
        if float(np.dot(q, qref_map[j])) < 0.0:
            q = -q
        qfix_map[j] = quat_normalize_wxyz(q.astype(np.float32))

    if not qfix_map:
        raise RuntimeError("failed to estimate per-joint quaternion fix map")
    print(f"[calib] quat_fix per_joint,right: n={len(qfix_map)}")

    # Sanity check: compute quat errors after applying fix
    if side == "right":
        csv_fixed = [apply_quat_right_multiply_per_joint(fr, qfix_map, default_qR_wxyz=None) for fr in csv_gmr]
    else:
        csv_fixed = [apply_quat_left_multiply_per_joint(fr, qfix_map, default_qL_wxyz=None) for fr in csv_gmr]
    # recompute FootMod for stability in BODY IK
    csv_fixed = [gmr_rename_and_footmod(fr, fmt=args.format) for fr in csv_fixed]

    angles = []
    for i in range(0, min(n_use, len(csv_fixed))):
        fb = bvh_gmr[best_off + i]
        fc = csv_fixed[i]
        for j in core:
            if j in fb and j in fc:
                angles.append(_ang_deg_between(fb[j][1], fc[j][1], quat_mul_wxyz=quat_mul_wxyz, quat_conj_wxyz=quat_conj_wxyz, quat_normalize_wxyz=quat_normalize_wxyz))
    if angles:
        print(f"[calib] quat_error_after_fix: mean={float(np.mean(angles)):.6f} deg max={float(np.max(angles)):.6f} deg (core joints)")

    # Build calib json
    out: Dict[str, Any] = {
        "version": 1,
        "source": {
            "csv_file": os.path.abspath(csv_path),
            "bvh_file": os.path.abspath(bvh_path),
            "format": str(args.format),
            "best_bvh_offset": int(best_off),
            "pos_mode": str(args.pos_mode),
        },
        "pos": {
            "units": "m",
            "apply_bvh_like_rotation": bool(apply_bvh_like),
        },
        "quat_fix": {
            "mode": "per_joint",
            "side": "right",
            "map_wxyz": {k: [float(x) for x in v.reshape(4)] for k, v in sorted(qfix_map.items())},
        },
        "notes": "Apply to CSV frames in order: pos.matrix (pos only) -> BVH_like_rotation (pos+quat) -> q'=q⊗q_fix_j. Recompute FootMod after quat fix for BODY IK stability.",
    }

    # Store as similarity (so rotation is applied to BOTH pos+quat at replay time).
    out["pos"]["s"] = float(pos_s)
    out["pos"]["R"] = [[float(x) for x in row] for row in pos_R.reshape(3, 3)]
    out["pos"]["t"] = [float(x) for x in pos_t.reshape(3)]
    out["notes"] = "Apply to CSV frames in order: pos.(s,R,t) (pos+quat) -> BVH_like_rotation (pos+quat) -> q'=q⊗q_fix_j. Recompute FootMod after quat fix for BODY IK stability."

    out_path = args.out_json
    if not os.path.isabs(out_path):
        out_path = os.path.join(repo, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ wrote calib json: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


