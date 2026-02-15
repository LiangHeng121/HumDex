#!/usr/bin/env python3
"""
Minimal CSV loader/exporter for XDMocap motionData_*.csv format.

CSV header example:
  time(ms), Hips position X(m), Hips position Y(m), Hips position Z(m),
           Hips quaternion W, Hips quaternion X, Hips quaternion Y, Hips quaternion Z, ...

Outputs:
  - joint_names: list[str]
  - pos_m: (T,J,3) float32
  - quat_wxyz: (T,J,4) float32

Also prints a small summary for debugging.

Dependencies: python3, numpy
"""

from __future__ import annotations

import argparse
import csv
import re
from typing import Dict, List, Tuple

import numpy as np


def infer_joints(fieldnames: List[str]) -> List[str]:
    names = set()
    for k in fieldnames:
        k = str(k).strip()
        # Examples: "Hips position X(m)", "LeftHand position Z(m)"
        m = re.match(r"^(.*)\s+position\s+[XYZ]\(m\)$", k)
        if m:
            names.add(m.group(1).strip())
            continue
        # Examples: "Hips quaternion W", "LeftHand quaternion Z"
        m = re.match(r"^(.*)\s+quaternion\s+[WXYZ]$", k)
        if m:
            names.add(m.group(1).strip())
            continue
    return sorted(names)


def getf(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def load_motiondata_csv(csv_file: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    with open(csv_file, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header")
        fieldnames = list(reader.fieldnames)
        joints = infer_joints(fieldnames)
        if not joints:
            raise RuntimeError("No joints inferred. Is this a motionData_*.csv?")

        times_ms: List[float] = []
        pos_list: List[np.ndarray] = []
        quat_list: List[np.ndarray] = []

        for row in reader:
            t = getf(row, "time(ms)", np.nan)
            times_ms.append(float(t))

            p = np.zeros((len(joints), 3), dtype=np.float32)
            q = np.zeros((len(joints), 4), dtype=np.float32)
            q[:, 0] = 1.0
            for j_idx, j in enumerate(joints):
                p[j_idx, 0] = getf(row, f"{j} position X(m)")
                p[j_idx, 1] = getf(row, f"{j} position Y(m)")
                p[j_idx, 2] = getf(row, f"{j} position Z(m)")
                q[j_idx, 0] = getf(row, f"{j} quaternion W", 1.0)
                q[j_idx, 1] = getf(row, f"{j} quaternion X", 0.0)
                q[j_idx, 2] = getf(row, f"{j} quaternion Y", 0.0)
                q[j_idx, 3] = getf(row, f"{j} quaternion Z", 0.0)

            pos_list.append(p)
            quat_list.append(q)

    times_ms_arr = np.asarray(times_ms, dtype=np.float64)
    pos = np.stack(pos_list, axis=0).astype(np.float32)
    quat = np.stack(quat_list, axis=0).astype(np.float32)
    return joints, times_ms_arr, pos, quat


def main() -> int:
    ap = argparse.ArgumentParser(description="Export motionData CSV pos/quat to npz")
    ap.add_argument("--csv_file", type=str, required=True)
    ap.add_argument("--out_npz", type=str, default="")
    args = ap.parse_args()

    joints, times_ms, pos, quat = load_motiondata_csv(args.csv_file)
    dt = np.diff(times_ms)
    fps = float("nan")
    if dt.size > 0:
        med = float(np.nanmedian(dt))
        if med > 1e-6:
            fps = 1000.0 / med

    print("======================================================================")
    print("motionData CSV export")
    print("======================================================================")
    print(f"csv_file: {args.csv_file}")
    print(f"frames  : {pos.shape[0]}")
    print(f"joints  : {pos.shape[1]}")
    print(f"fps_est : {fps}")
    print(f"first10 joints: {joints[:10]}")
    # show a few values for sanity
    for j in ["Hips", "Spine2", "LeftHand", "RightHand"]:
        if j in joints:
            idx = joints.index(j)
            print(f"[frame0] {j} pos={pos[0,idx].tolist()} quat_wxyz={quat[0,idx].tolist()}")

    if args.out_npz:
        np.savez(
            args.out_npz,
            joint_names=np.asarray(joints, dtype=object),
            times_ms=times_ms,
            pos_m=pos,
            quat_wxyz=quat,
        )
        print(f"wrote: {args.out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


