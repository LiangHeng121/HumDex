#!/usr/bin/env python3
"""
Minimal VMC (OSC) body viewer.

Listen to /VMC/Ext/Bone/Pos and visualize key body joints.
Dependencies: python-osc, numpy, matplotlib
"""
from __future__ import annotations

import argparse
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np  # type: ignore


def _normalize(name: str) -> str:
    return "".join([c.lower() for c in str(name) if c.isalnum()])


def _quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        q = (q / n).astype(np.float32)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _basis_matrix(swap: str, flip: str) -> np.ndarray:
    swap = str(swap).lower()
    flip = str(flip).lower()
    axes = {"x": 0, "y": 1, "z": 2}
    if len(swap) != 3 or any(c not in axes for c in swap):
        raise ValueError(f"invalid swap: {swap!r}")
    idx = [axes[c] for c in swap]
    B = np.eye(3, dtype=np.float32)[:, idx]
    f = np.ones(3, dtype=np.float32)
    for c in flip:
        if c in axes:
            f[axes[c]] *= -1.0
    B = B * f.reshape(1, 3)
    return B


def _parse_bvh_offsets(path: str) -> Tuple[Dict[str, Optional[str]], Dict[str, np.ndarray]]:
    parents: Dict[str, Optional[str]] = {}
    offsets: Dict[str, np.ndarray] = {}
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
                parent = None
                if stack:
                    parent = stack[-1]
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


def _build_fk_positions(
    raw_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    parents: Dict[str, Optional[str]],
    offsets: Dict[str, np.ndarray],
    bvh_to_vmc: Dict[str, str],
    scale: float,
    rot_mode: str,
    ref_global: Optional[Dict[str, np.ndarray]] = None,
    ref_local: Optional[Dict[str, np.ndarray]] = None,
    basis: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    pos_out: Dict[str, np.ndarray] = {}
    rot_out: Dict[str, np.ndarray] = {}
    rot_global: Dict[str, np.ndarray] = {}

    def get_vmc_rot(joint: str) -> np.ndarray:
        vmc_name = bvh_to_vmc.get(joint)
        if vmc_name is None:
            return np.eye(3, dtype=np.float32)
        key = _normalize(vmc_name)
        v = raw_map.get(key)
        if v is None:
            return np.eye(3, dtype=np.float32)
        _p, q = v
        r = _quat_to_mat_wxyz(q)
        if basis is not None:
            r = basis @ r @ basis.T
        return r

    def solve(joint: str) -> None:
        if joint in pos_out:
            return
        parent = parents.get(joint)
        if parent is None:
            r0 = get_vmc_rot(joint)
            rot_out[joint] = r0
            rot_global[joint] = r0
            pos_out[joint] = np.zeros(3, dtype=np.float32)
            return
        solve(parent)
        parent_rot = rot_out[parent]
        parent_global = rot_global[parent]
        r_vmc = get_vmc_rot(joint)
        if rot_mode == "global":
            if ref_global is not None and joint in ref_global:
                r_vmc = ref_global[joint].T @ r_vmc
            local_rot = parent_global.T @ r_vmc
            rot_global[joint] = r_vmc
        else:
            local_rot = r_vmc
            if ref_local is not None and joint in ref_local:
                local_rot = ref_local[joint].T @ local_rot
            rot_global[joint] = parent_global @ local_rot
        parent_pos = pos_out[parent]
        rot_out[joint] = parent_rot @ local_rot
        off = offsets.get(joint, np.zeros(3, dtype=np.float32))
        pos_out[joint] = parent_pos + parent_rot @ (off * float(scale))

    for j in parents.keys():
        solve(j)
    return pos_out


def _bvh_fk_to_vmc_pos(
    fk_pos: Dict[str, np.ndarray],
    bvh_to_vmc: Dict[str, str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for bvh_name, vmc_name in bvh_to_vmc.items():
        p = fk_pos.get(bvh_name)
        if p is None:
            continue
        out[_normalize(vmc_name)] = np.asarray(p, dtype=np.float32).reshape(3)
    return out


def _compute_ref_maps(
    raw_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    bvh_to_vmc: Dict[str, str],
    rot_mode: str,
    basis: Optional[np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    ref_g: Dict[str, np.ndarray] = {}
    ref_l: Dict[str, np.ndarray] = {}
    for bvh_name, vmc_name in bvh_to_vmc.items():
        key = _normalize(vmc_name)
        v = raw_map.get(key)
        if v is None:
            continue
        _p, q = v
        r = _quat_to_mat_wxyz(q)
        if basis is not None:
            r = basis @ r @ basis.T
        if rot_mode == "global":
            ref_g[bvh_name] = r
        else:
            ref_l[bvh_name] = r
    return ref_g, ref_l


class VmcReceiver:
    def __init__(self, ip: str, port: int) -> None:
        try:
            from pythonosc import dispatcher as osc_dispatcher  # type: ignore
            from pythonosc import osc_server  # type: ignore
        except Exception as e:
            raise SystemExit(
                "❌ 缺少依赖 `python-osc`。\n"
                "   解决：pip install python-osc\n"
                f"   原始错误：{e}"
            )

        self._lock = threading.Lock()
        self._bones: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._last_ts: float = 0.0
        self._seq: int = 0

        disp = osc_dispatcher.Dispatcher()
        disp.map("/VMC/Ext/Bone/Pos", self._on_bone_pos)

        self._server = osc_server.ThreadingOSCUDPServer((str(ip), int(port)), disp)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _on_bone_pos(
        self,
        _addr: str,
        name: str,
        px: float,
        py: float,
        pz: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> None:
        ts = time.time()
        pos = np.array([float(px), float(py), float(pz)], dtype=np.float32)
        quat = np.array([float(qw), float(qx), float(qy), float(qz)], dtype=np.float32)
        key = _normalize(name)
        with self._lock:
            self._bones[key] = (pos, quat)
            self._last_ts = float(ts)
            self._seq += 1

    def snapshot(self, max_age_s: float) -> Tuple[Optional[Dict[str, np.ndarray]], int, float]:
        with self._lock:
            now = time.time()
            if self._last_ts <= 0 or (now - float(self._last_ts)) > float(max_age_s):
                return None, -1, 0.0
            out: Dict[str, np.ndarray] = {}
            for k, (p, _q) in self._bones.items():
                out[k] = np.asarray(p, dtype=np.float32).reshape(3)
            return out, int(self._seq), float(self._last_ts)

    def snapshot_raw(self, max_age_s: float) -> Tuple[Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]], int, float]:
        with self._lock:
            now = time.time()
            if self._last_ts <= 0 or (now - float(self._last_ts)) > float(max_age_s):
                return None, -1, 0.0
            out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for k, (p, q) in self._bones.items():
                out[k] = (
                    np.asarray(p, dtype=np.float32).reshape(3),
                    np.asarray(q, dtype=np.float32).reshape(4),
                )
            return out, int(self._seq), float(self._last_ts)

    def close(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass


def _pick(pos_map: Dict[str, np.ndarray], names: list[str]) -> Optional[np.ndarray]:
    for n in names:
        v = pos_map.get(_normalize(n))
        if v is not None:
            return v
    return None


def _build_body_points(pos_map: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    def setp(name: str, cands: list[str], fallback: Optional[str] = None) -> None:
        v = _pick(pos_map, cands)
        if v is None and fallback is not None:
            v = out.get(fallback)
        if v is not None:
            out[name] = v

    setp("Hips", ["Hips", "Hip", "Root", "Pelvis"])
    setp("Spine", ["Spine", "Waist"], fallback="Hips")
    setp("Chest", ["Chest"], fallback="Spine")
    setp("UpperChest", ["UpperChest"], fallback="Chest")
    setp("Neck", ["Neck"], fallback="UpperChest")
    setp("Head", ["Head"], fallback="Neck")

    setp("RightShoulder", ["RightShoulder", "RightUpperShoulder"], fallback="UpperChest")
    setp("RightUpperArm", ["RightUpperArm", "RightArm"], fallback="RightShoulder")
    setp("RightLowerArm", ["RightLowerArm", "RightForeArm"], fallback="RightUpperArm")
    setp("RightHand", ["RightHand"], fallback="RightLowerArm")

    setp("LeftShoulder", ["LeftShoulder", "LeftUpperShoulder"], fallback="UpperChest")
    setp("LeftUpperArm", ["LeftUpperArm", "LeftArm"], fallback="LeftShoulder")
    setp("LeftLowerArm", ["LeftLowerArm", "LeftForeArm"], fallback="LeftUpperArm")
    setp("LeftHand", ["LeftHand"], fallback="LeftLowerArm")

    setp("RightUpperLeg", ["RightUpperLeg", "RightUpLeg", "RightThigh"], fallback="Hips")
    setp("RightLowerLeg", ["RightLowerLeg", "RightLeg", "RightCalf"], fallback="RightUpperLeg")
    setp("RightFoot", ["RightFoot", "RightAnkle"], fallback="RightLowerLeg")
    setp("RightToe", ["RightToe", "RightToes"], fallback="RightFoot")

    setp("LeftUpperLeg", ["LeftUpperLeg", "LeftUpLeg", "LeftThigh"], fallback="Hips")
    setp("LeftLowerLeg", ["LeftLowerLeg", "LeftLeg", "LeftCalf"], fallback="LeftUpperLeg")
    setp("LeftFoot", ["LeftFoot", "LeftAnkle"], fallback="LeftLowerLeg")
    setp("LeftToe", ["LeftToe", "LeftToes"], fallback="LeftFoot")

    return out


SKELETON_EDGES = [
    ("Hips", "Spine"),
    ("Spine", "Chest"),
    ("Chest", "UpperChest"),
    ("UpperChest", "Neck"),
    ("Neck", "Head"),
    ("UpperChest", "LeftShoulder"),
    ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftLowerArm"),
    ("LeftLowerArm", "LeftHand"),
    ("UpperChest", "RightShoulder"),
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


def main() -> int:
    p = argparse.ArgumentParser(description="Minimal VMC body viewer (OSC)")
    p.add_argument("--ip", type=str, default="0.0.0.0", help="OSC listen ip")
    p.add_argument("--port", type=int, default=39539, help="OSC listen port")
    p.add_argument("--timeout_s", type=float, default=0.5, help="Data timeout seconds")
    p.add_argument("--fps", type=float, default=60.0, help="UI refresh fps")
    p.add_argument("--scale", type=float, default=1.0, help="Position scale")
    p.add_argument("--show_all", action="store_true", help="Show all raw VMC points as gray dots")
    p.add_argument("--print_every", type=int, default=60, help="Print bone stats every N frames (0=off)")
    p.add_argument("--print_bones", action="store_true", help="Print raw bone names when logging")
    p.add_argument("--dump_once", action="store_true", help="Dump first frame bone pos/quat")
    p.add_argument("--dump_limit", type=int, default=30, help="Max bones to dump")
    p.add_argument("--use_fk", action="store_true", help="Reconstruct positions from quats + BVH offsets")
    p.add_argument("--bvh_path", type=str, default="bvh-recording.bvh", help="BVH file for offsets")
    p.add_argument("--bvh_scale", type=float, default=1.0, help="Scale BVH offsets")
    p.add_argument("--rot_mode", choices=["local", "global"], default="global", help="VMC rotations are local or global")
    p.add_argument("--use_ref_pose", action="store_true", help="Use first pose as reference to remove rest offset")
    p.add_argument("--ref_delay_s", type=float, default=1.0, help="Delay before capturing reference pose")
    p.add_argument("--axis_swap", type=str, default="xyz", help="Axis swap for VMC -> BVH (e.g. xzy, yxz)")
    p.add_argument("--axis_flip", type=str, default="", help="Axis flips for VMC -> BVH (e.g. x, yz)")
    p.add_argument("--axis_r", type=float, default=1.2, help="Auto-scale radius around hips")
    p.add_argument("--no_autoscale", action="store_true", help="Disable auto-scale around hips")
    p.add_argument("--fixed_axis", type=float, default=2.0, help="Fixed axis half-size when no_autoscale")
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise SystemExit(
            "❌ 缺少依赖 `matplotlib`。\n"
            "   解决：pip install matplotlib\n"
            f"   原始错误：{e}"
        )

    recv = VmcReceiver(str(args.ip), int(args.port))
    print(f"✅ Listening VMC on {args.ip}:{int(args.port)}")

    parents: Dict[str, Optional[str]] = {}
    offsets: Dict[str, np.ndarray] = {}
    bvh_to_vmc: Dict[str, str] = {}
    ref_global: Optional[Dict[str, np.ndarray]] = None
    ref_local: Optional[Dict[str, np.ndarray]] = None
    ref_ready = False
    t_start = time.time()
    basis = None
    try:
        basis = _basis_matrix(str(args.axis_swap), str(args.axis_flip))
    except Exception as e:
        print(f"⚠️ 轴变换参数无效，使用默认 xyz：{e}")
        basis = _basis_matrix("xyz", "")
    if bool(args.use_fk):
        parents, offsets = _parse_bvh_offsets(str(args.bvh_path))
        # Map BVH joint names -> VMC bone names (based on your recording)
        bvh_to_vmc = {
            "HIP": "Hips",
            "WAIST": "Spine",
            "CHEST": "Chest",
            "UPPER_CHEST": "UpperChest",
            "NECK": "Neck",
            "HEAD": "Head",
            "LEFT_UPPER_SHOULDER": "LeftShoulder",
            "LEFT_SHOULDER": "LeftShoulder",
            "LEFT_UPPER_ARM": "LeftUpperArm",
            "LEFT_LOWER_ARM": "LeftLowerArm",
            "LEFT_HAND": "LeftHand",
            "RIGHT_UPPER_SHOULDER": "RightShoulder",
            "RIGHT_SHOULDER": "RightShoulder",
            "RIGHT_UPPER_ARM": "RightUpperArm",
            "RIGHT_LOWER_ARM": "RightLowerArm",
            "RIGHT_HAND": "RightHand",
            "LEFT_HIP": "Hips",
            "LEFT_UPPER_LEG": "LeftUpperLeg",
            "LEFT_LOWER_LEG": "LeftLowerLeg",
            "LEFT_FOOT": "LeftFoot",
            "LEFT_TOE": "LeftFoot",
            "RIGHT_HIP": "Hips",
            "RIGHT_UPPER_LEG": "RightUpperLeg",
            "RIGHT_LOWER_LEG": "RightLowerLeg",
            "RIGHT_FOOT": "RightFoot",
            "RIGHT_TOE": "RightFoot",
        }

    plt.ion()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20.0, azim=-60.0)

    scat = ax.scatter([], [], [], s=15)
    scat_raw = ax.scatter([], [], [], s=6, c="gray", alpha=0.5)
    line_objs = [ax.plot([], [], [], lw=2)[0] for _ in SKELETON_EDGES]

    last_seq = -1
    dumped = False
    step = 0
    try:
        while True:
            if bool(args.use_fk):
                raw_map, seq, _ts = recv.snapshot_raw(float(args.timeout_s))
                if raw_map is None:
                    time.sleep(0.01)
                    plt.pause(0.001)
                    continue
                pos_map = {k: v[0] for k, v in raw_map.items()}
            else:
                pos_map, seq, _ts = recv.snapshot(float(args.timeout_s))
                raw_map = None
                if pos_map is None:
                    time.sleep(0.01)
                    plt.pause(0.001)
                    continue
            if pos_map is None:
                time.sleep(0.01)
                plt.pause(0.001)
                continue
            if seq == last_seq:
                time.sleep(0.001)
                plt.pause(0.001)
                continue
            last_seq = seq

            if bool(args.use_fk) and raw_map is not None:
                if bool(args.use_ref_pose) and (not ref_ready) and (time.time() - t_start >= float(args.ref_delay_s)):
                    ref_global, ref_local = _compute_ref_maps(raw_map, bvh_to_vmc, str(args.rot_mode), basis)
                    ref_ready = True
                    print("✅ 已捕获参考姿态（ref pose）")
                fk_pos = _build_fk_positions(
                    raw_map,
                    parents,
                    offsets,
                    bvh_to_vmc,
                    float(args.bvh_scale),
                    str(args.rot_mode),
                    ref_global=ref_global,
                    ref_local=ref_local,
                    basis=basis,
                )
                pos_map_vmc = _bvh_fk_to_vmc_pos(fk_pos, bvh_to_vmc)
                pts = _build_body_points(pos_map_vmc)
            else:
                pts = _build_body_points(pos_map)
            if not pts:
                time.sleep(0.01)
                plt.pause(0.001)
                continue

            xyz = np.stack([v for v in pts.values()], axis=0) * float(args.scale)
            xs, ys, zs = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            scat._offsets3d = (xs, ys, zs)

            if bool(args.show_all):
                show_map = pos_map
                if bool(args.use_fk) and raw_map is not None:
                    show_map = pos_map_vmc
                raw_xyz = np.stack([v for v in show_map.values()], axis=0) * float(args.scale)
                rxs, rys, rzs = raw_xyz[:, 0], raw_xyz[:, 1], raw_xyz[:, 2]
                scat_raw._offsets3d = (rxs, rys, rzs)
            else:
                scat_raw._offsets3d = ([], [], [])

            for i, (a, b) in enumerate(SKELETON_EDGES):
                pa = pts.get(a)
                pb = pts.get(b)
                if pa is None or pb is None:
                    line_objs[i].set_data([], [])
                    line_objs[i].set_3d_properties([])
                    continue
                p1 = pa * float(args.scale)
                p2 = pb * float(args.scale)
                line_objs[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                line_objs[i].set_3d_properties([p1[2], p2[2]])

            # auto-scale around hips
            if not bool(args.no_autoscale):
                hips = pts.get("Hips")
                if hips is not None:
                    h = hips * float(args.scale)
                    r = float(args.axis_r)
                    ax.set_xlim(h[0] - r, h[0] + r)
                    ax.set_ylim(h[1] - r, h[1] + r)
                    ax.set_zlim(h[2] - r, h[2] + r)
            else:
                r = float(args.fixed_axis)
                ax.set_xlim(-r, r)
                ax.set_ylim(-r, r)
                ax.set_zlim(-r, r)

            fig.canvas.draw_idle()
            plt.pause(0.001)
            step += 1
            if int(args.print_every) > 0 and (step % int(args.print_every) == 0):
                raw_count = len(pos_map)
                body_count = len(pts)
                print(f"[vmc] raw_bones={raw_count} mapped_body={body_count}")
                if raw_count > 0:
                    raw_xyz = np.stack([v for v in pos_map.values()], axis=0)
                    raw_min = raw_xyz.min(axis=0)
                    raw_max = raw_xyz.max(axis=0)
                    raw_span = raw_max - raw_min
                    print(f"  raw_span: x={raw_span[0]:.4f} y={raw_span[1]:.4f} z={raw_span[2]:.4f}")
                if body_count > 0:
                    body_xyz = np.stack([v for v in pts.values()], axis=0)
                    body_min = body_xyz.min(axis=0)
                    body_max = body_xyz.max(axis=0)
                    body_span = body_max - body_min
                    print(f"  body_span: x={body_span[0]:.4f} y={body_span[1]:.4f} z={body_span[2]:.4f}")
                if bool(args.use_fk) and raw_map is not None and body_count == 0:
                    print("  ⚠️ FK 位置映射为空，检查 BVH 名称映射是否匹配。")
                if bool(args.print_bones):
                    keys = sorted(pos_map.keys())
                    print("  raw_names:", ", ".join(keys[:80]))
                    if len(keys) > 80:
                        print("  ...")
                if body_count <= 1:
                    print("  ⚠️ 映射后只有一个点，检查 VMC 骨骼命名或发送设置。")

            if bool(args.dump_once) and (not dumped):
                raw_map, _seq2, _ts2 = recv.snapshot_raw(float(args.timeout_s))
                if raw_map is not None:
                    dumped = True
                    keys = sorted(raw_map.keys())
                    print(f"[dump] total={len(keys)}")
                    for k in keys[: int(args.dump_limit)]:
                        p, q = raw_map[k]
                        print(f"  {k}: pos=({p[0]:.6f},{p[1]:.6f},{p[2]:.6f}) quat=({q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f})")

            time.sleep(max(0.0, 1.0 / max(1.0, float(args.fps))))
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，退出。")
    finally:
        recv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

