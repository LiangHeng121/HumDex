#!/usr/bin/env python3
"""
Minimal BVH FK loader/exporter (same logic as our replay BVH loader).

What it does:
1) read BVH
2) FK to global pos+quat (wxyz)
3) apply fixed axis transform used in our pipeline:
     rotation_matrix = [[1,0,0],[0,0,-1],[0,1,0]]
     pos: p' = p @ rotation_matrix.T / 100.0   (cm -> m)
     quat: q' = rotation_quat ⊗ q
4) apply small name aliases to match common naming (Upper/Lower -> Up/Fore etc.)

Outputs:
  - bone_names: list[str]
  - pos_m: (T,J,3) float32
  - quat_wxyz: (T,J,4) float32

Dependencies (same as current repo pipeline):
  - numpy
  - scipy
  - general_motion_retargeting.utils.lafan_vendor (comes with this repo / gmr env)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore


def _ensure_gmr_on_path() -> None:
    # allow running from anywhere inside this repo
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gmr_root = os.path.join(repo_root, "GMR")
    if gmr_root not in sys.path:
        sys.path.insert(0, gmr_root)


def _rename_aliases(result: Dict[str, Any]) -> Dict[str, Any]:
    rename = {
        "LeftUpperLeg": "LeftUpLeg",
        "RightUpperLeg": "RightUpLeg",
        "LeftLowerLeg": "LeftLeg",
        "RightLowerLeg": "RightLeg",
        "LeftUpperArm": "LeftArm",
        "RightUpperArm": "RightArm",
        "LeftLowerArm": "LeftForeArm",
        "RightLowerArm": "RightForeArm",
    }
    out = dict(result)
    for src, dst in rename.items():
        if src in out and dst not in out:
            out[dst] = out[src]
    # Toe / ToeBase aliases
    if "LeftToe" in out and "LeftToeBase" not in out:
        out["LeftToeBase"] = out["LeftToe"]
    if "RightToe" in out and "RightToeBase" not in out:
        out["RightToeBase"] = out["RightToe"]
    if "LeftToeBase" in out and "LeftToe" not in out:
        out["LeftToe"] = out["LeftToeBase"]
    if "RightToeBase" in out and "RightToe" not in out:
        out["RightToe"] = out["RightToeBase"]
    return out


def load_bvh_fk(bvh_file: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    _ensure_gmr_on_path()
    import general_motion_retargeting.utils.lafan_vendor.utils as utils  # type: ignore
    from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh  # type: ignore

    data = read_bvh(bvh_file)
    global_q, global_p = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True).astype(np.float32)

    T = data.pos.shape[0]
    bones = list(data.bones)

    # export with aliases merged into a canonical dict per frame, then re-pack to arrays by sorted joint names
    frames: List[Dict[str, Any]] = []
    for f in range(T):
        fr: Dict[str, Any] = {}
        for i, name in enumerate(bones):
            q = utils.quat_mul(rotation_quat, global_q[f, i])
            p = global_p[f, i] @ rotation_matrix.T / 100.0  # cm -> m
            fr[name] = [np.asarray(p, dtype=np.float32), np.asarray(q, dtype=np.float32)]
        fr = _rename_aliases(fr)
        frames.append(fr)

    # create a stable joint list (sorted)
    joint_names = sorted(list(frames[0].keys()))
    pos = np.zeros((T, len(joint_names), 3), dtype=np.float32)
    quat = np.zeros((T, len(joint_names), 4), dtype=np.float32)
    quat[:, :, 0] = 1.0
    for ti, fr in enumerate(frames):
        for ji, j in enumerate(joint_names):
            if j in fr:
                pos[ti, ji] = np.asarray(fr[j][0], dtype=np.float32).reshape(3)
                quat[ti, ji] = np.asarray(fr[j][1], dtype=np.float32).reshape(4)
    return joint_names, pos, quat


def main() -> int:
    ap = argparse.ArgumentParser(description="Export BVH FK global pos/quat to npz")
    ap.add_argument("--bvh_file", type=str, required=True)
    ap.add_argument("--out_npz", type=str, default="")
    args = ap.parse_args()

    joint_names, pos, quat = load_bvh_fk(args.bvh_file)
    print("======================================================================")
    print("BVH FK export")
    print("======================================================================")
    print(f"bvh_file: {args.bvh_file}")
    print(f"frames  : {pos.shape[0]}")
    print(f"joints  : {pos.shape[1]}")
    print(f"first10 joints: {joint_names[:10]}")
    for j in ["Hips", "Spine2", "LeftHand", "RightHand"]:
        if j in joint_names:
            idx = joint_names.index(j)
            print(f"[frame0] {j} pos={pos[0,idx].tolist()} quat_wxyz={quat[0,idx].tolist()}")

    if args.out_npz:
        np.savez(
            args.out_npz,
            joint_names=np.asarray(joint_names, dtype=object),
            pos_m=pos,
            quat_wxyz=quat,
        )
        print(f"wrote: {args.out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


