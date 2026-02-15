#!/usr/bin/env python3
"""
Estimate a per-joint constant LOCAL rotation correction from CSV quats to BVH(FK) quats,
assuming the motion is the same and positions can be aligned with similarity + time offset.

Why local? Many pipelines differ by bone local coordinate frames (pre-rotations). A constant
correction is usually defined in LOCAL (parent) space:

    Rloc_bvh(t) ~= Rcorr_j * Rloc_csv(t)

We solve Rcorr_j by averaging relative rotations:
    Rrel_j(t) = Rloc_bvh(t) * Rloc_csv(t)^T

Then apply correction and report residual angle errors.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


CORE = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "RightToeBase",
]

_QUTILS = None


def _qutils():
    """
    Lazy import quaternion/similarity helpers from pose_csv_loader after repo path is ready.
    """
    global _QUTILS
    if _QUTILS is not None:
        return _QUTILS
    _ensure_repo_on_path()
    from deploy_real.pose_csv_loader import (  # type: ignore
        umeyama_similarity_transform,
        apply_similarity_transform_frame,
        quat_normalize_wxyz,
        quat_mul_wxyz,
        quat_conj_wxyz,
        quat_wxyz_to_rotmat,
        rotmat_to_quat_wxyz,
    )

    _QUTILS = {
        "umeyama_similarity_transform": umeyama_similarity_transform,
        "apply_similarity_transform_frame": apply_similarity_transform_frame,
        "quat_normalize_wxyz": quat_normalize_wxyz,
        "quat_mul_wxyz": quat_mul_wxyz,
        "quat_conj_wxyz": quat_conj_wxyz,
        "quat_wxyz_to_rotmat": quat_wxyz_to_rotmat,
        "rotmat_to_quat_wxyz": rotmat_to_quat_wxyz,
    }
    return _QUTILS


def parent_map_gmr() -> Dict[str, Optional[str]]:
    p: Dict[str, Optional[str]] = {}
    p["Hips"] = None
    # spine chain
    p["Spine"] = "Hips"
    p["Spine1"] = "Spine"
    p["Spine2"] = "Spine1"
    p["Spine3"] = "Spine2"
    p["Neck"] = "Spine3"
    p["Head"] = "Neck"
    # legs
    p["LeftUpLeg"] = "Hips"
    p["LeftLeg"] = "LeftUpLeg"
    p["LeftFoot"] = "LeftLeg"
    p["LeftToeBase"] = "LeftFoot"
    p["LeftToe"] = "LeftFoot"  # alias; some data puts it under foot directly
    p["RightUpLeg"] = "Hips"
    p["RightLeg"] = "RightUpLeg"
    p["RightFoot"] = "RightLeg"
    p["RightToeBase"] = "RightFoot"
    p["RightToe"] = "RightFoot"
    # arms
    p["LeftShoulder"] = "Spine3"
    p["LeftArm"] = "LeftShoulder"
    p["LeftForeArm"] = "LeftArm"
    p["LeftHand"] = "LeftForeArm"
    p["RightShoulder"] = "Spine3"
    p["RightArm"] = "RightShoulder"
    p["RightForeArm"] = "RightArm"
    p["RightHand"] = "RightForeArm"
    return p


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


def _load_csv_frames(csv_file: str, fmt: str) -> List[Dict[str, Any]]:
    _ensure_repo_on_path()
    from deploy_real.pose_csv_loader import load_pose_csv_frames, gmr_rename_and_footmod  # type: ignore

    raw, _meta = load_pose_csv_frames(csv_file, include_body=True, include_lhand=False, include_rhand=False)
    frames = [gmr_rename_and_footmod(fr, fmt=fmt) for fr in raw]
    return frames


def _quat_wxyz(fr: Dict[str, Any], j: str) -> Optional[np.ndarray]:
    if j not in fr:
        return None
    v = fr[j]
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return None
    return np.asarray(v[1], dtype=np.float32).reshape(4)


def _pos_xyz(fr: Dict[str, Any], j: str) -> Optional[np.ndarray]:
    if j not in fr:
        return None
    v = fr[j]
    if not isinstance(v, (list, tuple)) or len(v) < 1:
        return None
    return np.asarray(v[0], dtype=np.float32).reshape(3)


def mean_quat_wxyz(quats: List[np.ndarray]) -> np.ndarray:
    # Markley average
    qn = _qutils()["quat_normalize_wxyz"]
    A = np.zeros((4, 4), dtype=np.float64)
    for q in quats:
        q = qn(q)
        if q[0] < 0:
            q = -q
        A += np.outer(q, q)
    A /= max(1, len(quats))
    w, V = np.linalg.eigh(A)
    q = V[:, int(np.argmax(w))]
    q = np.asarray(q, dtype=np.float32)
    if q[0] < 0:
        q = -q
    return qn(q)


def ang_deg(q: np.ndarray) -> float:
    q = _qutils()["quat_normalize_wxyz"](q)
    w = float(np.clip(q[0], -1.0, 1.0))
    return float(2.0 * np.arccos(abs(w)) * 180.0 / np.pi)


def fit_similarity_and_offset(
    csv_frames: List[Dict[str, Any]],
    bvh_frames: List[Dict[str, Any]],
    joints: List[str],
    frames: int,
    stride: int,
    search_offset: int,
) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """
    Return: best_off, s, Rm, t
    """
    best = None  # (pos_rms, off, s, Rm, t)
    umeyama_similarity_transform = _qutils()["umeyama_similarity_transform"]
    for off in range(-int(search_offset), int(search_offset) + 1):
        bs = off
        if bs < 0:
            continue
        max_n = min(len(csv_frames), len(bvh_frames) - bs)
        n_use = max_n if int(frames) < 0 else min(max_n, int(frames))
        xs, ys = [], []
        for i in range(0, n_use, max(1, int(stride))):
            fc = csv_frames[i]
            fb = bvh_frames[bs + i]
            for j in joints:
                pc = _pos_xyz(fc, j)
                pb = _pos_xyz(fb, j)
                if pc is None or pb is None:
                    continue
                xs.append(pc)
                ys.append(pb)
        if len(xs) < 3:
            continue
        X = np.stack(xs, axis=0)
        Y = np.stack(ys, axis=0)
        s, Rm, t = umeyama_similarity_transform(X, Y, with_scale=True)
        Yhat = (float(s) * (X @ Rm.T)) + t.reshape(1, 3)
        resid = Y - Yhat
        rms = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
        cand = (rms, off, float(s), np.asarray(Rm, dtype=np.float32), np.asarray(t, dtype=np.float32))
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        raise RuntimeError("failed to fit similarity/offset")
    _rms, best_off, s, Rm, t = best
    return int(best_off), float(s), Rm, t


def compute_local_rots(fr: Dict[str, Any], parents: Dict[str, Optional[str]]) -> Dict[str, np.ndarray]:
    """
    Return dict joint->Rloc (3x3)
    """
    out: Dict[str, np.ndarray] = {}
    quat_wxyz_to_rotmat = _qutils()["quat_wxyz_to_rotmat"]
    for j, pj in parents.items():
        if j not in fr:
            continue
        qg = _quat_wxyz(fr, j)
        if qg is None:
            continue
        Rg = quat_wxyz_to_rotmat(qg)
        if pj is None or pj not in fr:
            out[j] = Rg
        else:
            qpg = _quat_wxyz(fr, pj)
            if qpg is None:
                out[j] = Rg
            else:
                Rpg = quat_wxyz_to_rotmat(qpg)
                out[j] = (Rpg.T @ Rg).astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate per-joint local quat mapping CSV -> BVH")
    ap.add_argument("--csv_file", type=str, required=True)
    ap.add_argument("--bvh_file", type=str, required=True)
    ap.add_argument("--format", choices=["lafan1", "nokov"], default="nokov")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--search_time_offset", type=int, default=20)
    ap.add_argument("--out_json", type=str, default="", help="Write per-joint qcorr (wxyz) to json")
    args = ap.parse_args()

    csv_path = _resolve_path(args.csv_file)
    bvh_path = _resolve_path(args.bvh_file)
    if not os.path.exists(csv_path):
        print(f"❌ 找不到 CSV: {args.csv_file}")
        return 2
    if not os.path.exists(bvh_path):
        print(f"❌ 找不到 BVH: {args.bvh_file}")
        return 2

    csv_frames = _load_csv_frames(csv_path, fmt=args.format)
    bvh_frames = _load_bvh_frames(bvh_path, fmt=args.format)

    parents = parent_map_gmr()
    # use joints present in both
    pos_joints = [j for j in CORE if j in csv_frames[0] and j in bvh_frames[0] and not j.endswith("FootMod")]

    apply_similarity_transform_frame = _qutils()["apply_similarity_transform_frame"]
    quat_mul_wxyz = _qutils()["quat_mul_wxyz"]
    quat_conj_wxyz = _qutils()["quat_conj_wxyz"]
    quat_wxyz_to_rotmat = _qutils()["quat_wxyz_to_rotmat"]
    rotmat_to_quat_wxyz = _qutils()["rotmat_to_quat_wxyz"]

    best_off, s, Rm, t = fit_similarity_and_offset(
        csv_frames,
        bvh_frames,
        joints=pos_joints,
        frames=int(args.frames),
        stride=int(args.stride),
        search_offset=int(args.search_time_offset),
    )
    print("======================================================================")
    print("Estimate LOCAL per-joint quat correction (CSV -> BVH)")
    print("======================================================================")
    print(f"best_off: {best_off}")
    print(f"pos_fit : s={s:.8f}, det(R)={float(np.linalg.det(Rm)):.3f}, t={t.tolist()}")

    # apply similarity to CSV frames (rotates quats by qR ⊗ q)
    csv_aligned = [apply_similarity_transform_frame(fr, s, Rm, t) for fr in csv_frames]

    # fit per-joint local correction
    qcorr: Dict[str, np.ndarray] = {}
    for j, pj in parents.items():
        if j not in csv_aligned[0] or j not in bvh_frames[best_off]:
            continue
        rels: List[np.ndarray] = []
        for fi in range(0, min(int(args.frames) if int(args.frames) > 0 else 240, len(csv_aligned) - 1), int(args.stride)):
            fb = bvh_frames[best_off + fi]
            fc = csv_aligned[fi]
            # compute local quats via Rloc = Rparent^T R
            # qrel = qloc_bvh ⊗ inv(qloc_csv)
            # We'll compute via rotation matrices to avoid ambiguity.
            # Skip if missing parent.
            def Rloc(fr, joint, parent):
                qg = _quat_wxyz(fr, joint)
                if qg is None:
                    return None
                Rg = quat_wxyz_to_rotmat(qg)
                if parent is None or parent not in fr:
                    return Rg
                qpg = _quat_wxyz(fr, parent)
                if qpg is None:
                    return Rg
                Rpg = quat_wxyz_to_rotmat(qpg)
                return (Rpg.T @ Rg).astype(np.float32)

            Rb = Rloc(fb, j, pj)
            Rc = Rloc(fc, j, pj)
            if Rb is None or Rc is None:
                continue
            Rrel = (Rb @ Rc.T).astype(np.float32)
            # convert to quat to average
            # We can get quat via rotmat_to_quat from pose_csv_loader, import locally.
            rels.append(rotmat_to_quat_wxyz(Rrel))
        if len(rels) >= 3:
            qcorr[j] = mean_quat_wxyz(rels)

    # score residuals in local space
    all_ang: List[float] = []
    per: Dict[str, Tuple[float, float, int]] = {}
    for j, pj in parents.items():
        if j not in qcorr:
            continue
        angs: List[float] = []
        for fi in range(0, min(240, len(csv_aligned) - 1), int(args.stride)):
            fb = bvh_frames[best_off + fi]
            fc = csv_aligned[fi]
            def qloc(fr, joint, parent):
                qg = _quat_wxyz(fr, joint)
                if qg is None:
                    return None
                if parent is None or parent not in fr:
                    return qg
                qpg = _quat_wxyz(fr, parent)
                if qpg is None:
                    return qg
                # qloc = inv(qparent) ⊗ qglob
                return quat_mul_wxyz(quat_conj_wxyz(qpg), qg)

            qb = qloc(fb, j, pj)
            qc = qloc(fc, j, pj)
            if qb is None or qc is None:
                continue
            qpred = quat_mul_wxyz(qcorr[j], qc)
            qerr = quat_mul_wxyz(qb, quat_conj_wxyz(qpred))
            a = ang_deg(qerr)
            angs.append(a)
            all_ang.append(a)
        if angs:
            per[j] = (float(np.mean(angs)), float(np.max(angs)), int(len(angs)))

    mean_all = float(np.mean(all_ang)) if all_ang else float("inf")
    max_all = float(np.max(all_ang)) if all_ang else float("inf")
    print(f"local_quat_residual: mean={mean_all:.3f} deg, max={max_all:.3f} deg, joints={len(per)}")
    worst = sorted(per.items(), key=lambda kv: kv[1][0], reverse=True)[:10]
    print("worst joints (mean/max deg):")
    for j, (m, ma, n) in worst:
        print(f"  {j:>14s}: mean={m:.3f}, max={ma:.3f}, n={n}")

    if args.out_json:
        out = {j: q.tolist() for j, q in qcorr.items()}
        with open(args.out_json, "w", encoding="utf-8") as f:
            import json
            json.dump(
                {
                    "best_off": int(best_off),
                    "pos_fit": {"s": float(s), "R": Rm.tolist(), "t": t.tolist()},
                    "qcorr_local_wxyz": out,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


