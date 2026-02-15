#!/usr/bin/env python3
"""
Replay 3D FK pose data (CSV) to G1 via Redis (BODY ONLY).

目标：
- 读取“FK 后的 3D 关节位姿 CSV”（包含 body_* 的 px/py/pz + qw/qx/qy/qz）
  并通过 GMR 做全身重定向
- 按固定 FPS 循环写 Redis：
  - action_body_unitree_g1_with_hands (35D)
  - action_hand_left/right_unitree_g1_with_hands (7D) -> 固定全 0
  - action_neck_unitree_g1_with_hands (2D) -> 固定 [0,0]
  - t_action (ms)
- 不控制 Wuji：默认把 wuji_hand_mode_{left/right}_unitree_g1_with_hands 置为 default，
  并把 hand_tracking_{left/right}_unitree_g1_with_hands 置为 is_active=false。

使用前提：
- 本机 Redis 正在运行（通常 localhost:6379）
- sim2real.sh 已启动（低层在读 action_* 并执行）
"""

import argparse
import json
import os
import sys
import time
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import redis


def _now_ms() -> int:
    return int(time.time() * 1000)


class _RateLimiter:
    def __init__(self, hz: float):
        self.dt = 1.0 / max(1e-6, float(hz))
        self.next_t = time.monotonic()

    def sleep(self):
        self.next_t += self.dt
        now = time.monotonic()
        delay = self.next_t - now
        if delay > 0:
            time.sleep(delay)
        else:
            # 如果落后太多，重置避免“越睡越慢”
            self.next_t = now


def _import_gmr() -> Tuple[object, object]:
    """
    Return (GMR_class, bvh_loader_func).
    兼容用户没有把 GMR 安装进 site-packages 的情况：尝试把仓库内 GMR 加到 sys.path。
    """
    try:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR  # type: ignore
        return GMR, _load_bvh_frames_via_gmr
    except Exception:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        gmr_root = os.path.join(repo_root, "GMR")
        if gmr_root not in sys.path:
            sys.path.insert(0, gmr_root)
        from general_motion_retargeting import GeneralMotionRetargeting as GMR  # type: ignore
        return GMR, _load_bvh_frames_via_gmr


def _load_bvh_frames_via_gmr(bvh_file: str, fmt: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    不依赖 general_motion_retargeting/utils/lafan1.py（避免 LeftToeBase / 命名差异导致的 KeyError），
    直接读取 BVH 并构造 GMR 需要的 frames。

    - fmt=lafan1: LeftFootMod/RightFootMod 优先用 Toe（没有则用 ToeBase）
    - fmt=nokov : LeftFootMod/RightFootMod 优先用 ToeBase（没有则用 Toe）
    - 同时做常见骨骼命名映射（Upper/Lower -> Up/Fore 等）
    """
    # Ensure GMR is importable (sys.path handled by _import_gmr caller)
    import general_motion_retargeting.utils.lafan_vendor.utils as utils  # type: ignore
    from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh  # type: ignore
    from scipy.spatial.transform import Rotation as R  # type: ignore

    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    # Match original lafan1.py coordinate convention
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    def _rename(result: Dict[str, Any]) -> Dict[str, Any]:
        # Upper/Lower naming variants -> IK config naming
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
        # Toe / ToeBase 双向别名（便于 nokov/lafan1 两种 config）
        if "LeftToe" in out and "LeftToeBase" not in out:
            out["LeftToeBase"] = out["LeftToe"]
        if "RightToe" in out and "RightToeBase" not in out:
            out["RightToeBase"] = out["RightToe"]
        if "LeftToeBase" in out and "LeftToe" not in out:
            out["LeftToe"] = out["LeftToeBase"]
        if "RightToeBase" in out and "RightToe" not in out:
            out["RightToe"] = out["RightToeBase"]
        return out

    frames: List[Dict[str, Any]] = []
    for frame in range(data.pos.shape[0]):
        result: Dict[str, Any] = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100.0  # cm -> m
            result[bone] = [position, orientation]

        result = _rename(result)

        # FootMod selection depending on format/config
        if fmt == "lafan1":
            # prefer Toe orientation, fallback ToeBase
            left_toe = "LeftToe" if "LeftToe" in result else "LeftToeBase"
            right_toe = "RightToe" if "RightToe" in result else "RightToeBase"
            result["LeftFootMod"] = [result["LeftFoot"][0], result[left_toe][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result[right_toe][1]]
        elif fmt == "nokov":
            # prefer ToeBase orientation, fallback Toe
            left_toe = "LeftToeBase" if "LeftToeBase" in result else "LeftToe"
            right_toe = "RightToeBase" if "RightToeBase" in result else "RightToe"
            result["LeftFootMod"] = [result["LeftFoot"][0], result[left_toe][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result[right_toe][1]]
        else:
            raise ValueError(f"Invalid format: {fmt}")

        frames.append(result)

    # 与原 lafan1.py 一致：先用固定值（不影响重定向的“骨架相对比例”太多）
    human_height = 1.75
    return frames, human_height


def _load_pose_csv_frames(csv_file: str, fmt: str) -> Tuple[List[Dict[str, Any]], float, Optional[float]]:
    """
    Load FK-ed pose CSV and convert to frames compatible with GMR input.
    Returns (frames, human_height, fps_from_csv_or_None).
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from deploy_real.pose_csv_loader import load_pose_csv_frames, gmr_rename_and_footmod  # type: ignore

    frames_raw, meta = load_pose_csv_frames(
        csv_file, include_body=True, include_lhand=False, include_rhand=False, max_frames=-1
    )
    frames = [gmr_rename_and_footmod(fr, fmt=fmt) for fr in frames_raw]
    human_height = 1.75
    return frames, human_height, meta.fps


def extract_mimic_obs_whole_body(qpos: np.ndarray, last_qpos: np.ndarray, dt: float) -> np.ndarray:
    """
    与 teleop 中保持一致：从 qpos 提取 35D mimic_obs
    """
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
            base_vel_local[:2],  # xy vel
            root_pos[2:3],  # z
            roll,
            pitch,
            base_ang_vel_local[2:3],  # yaw rate
            robot_joints,  # 29
        ]
    )
    return mimic_obs


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay 3D FK pose CSV to G1 via Redis (body only)")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to pose CSV (FK-ed 3D joints)")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1", help="BVH format")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis IP")
    parser.add_argument("--fps", type=float, default=30.0, help="Replay FPS (default: 30). If --use_csv_fps, overrides from CSV frequency")
    parser.add_argument("--use_csv_fps", action="store_true", help="Use FPS from CSV 'frequency' column if available")
    parser.add_argument("--csv_calib_json", type=str, default="", help="Apply a pre-derived CSV calibration json (so replay does NOT need a BVH file). This is applied before other CSV transforms.")
    parser.add_argument("--csv_units", choices=["m", "cm", "mm"], default="m", help="Units of CSV position fields (default: m)")
    parser.add_argument("--csv_apply_bvh_rotation", action="store_true", help="Apply BVH->GMR fixed axis rotation to CSV pos/quat (convert BVH raw world -> GMR world)")
    parser.add_argument("--csv_geo_to_bvh_official", action="store_true", help="Apply vendor official Geo->BVH mapping to CSV pos+quat: pos(-x,z,y), quat(w,-x,z,y). Intended for BVH raw world.")
    parser.add_argument("--csv_geo_to_bvh_official_pos_only", action="store_true", help="Apply vendor official Geo->BVH mapping to CSV positions only: pos(-x,z,y), keep quats unchanged.")
    parser.add_argument("--csv_quat_order", choices=["wxyz", "xyzw"], default="wxyz", help="Quaternion order in CSV (default: wxyz)")
    parser.add_argument("--csv_quat_space", choices=["global", "local"], default="global", help="Quaternion space in CSV (default: global)")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--start", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (exclusive), -1 for end")
    parser.add_argument("--offset_to_ground", action="store_true", help="Offset human motion to ground")
    parser.add_argument("--print_every", type=int, default=60, help="Print status every N frames")
    parser.add_argument("--dry_run", action="store_true", help="Do not write Redis, only run retarget")
    args = parser.parse_args()

    GMR, load_bvh_file = _import_gmr()

    # Resolve CSV path: wrapper 可能 cd 到 deploy_real，用户常给的是仓库根目录相对路径
    csv_path = args.csv_file
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cand = os.path.join(repo_root, csv_path)
        if os.path.exists(cand):
            csv_path = cand
    if not os.path.exists(csv_path):
        print(f"❌ 找不到 CSV 文件: {args.csv_file}", file=sys.stderr)
        print("   你可以传绝对路径，或在仓库根目录运行脚本。", file=sys.stderr)
        return 2

    # Load pose frames (human data dict list)
    frames, actual_human_height, fps_from_csv = _load_pose_csv_frames(csv_path, args.format)
    if (
        bool(getattr(args, "csv_calib_json", ""))
        or
        args.csv_quat_order != "wxyz"
        or args.csv_quat_space != "global"
        or bool(getattr(args, "csv_geo_to_bvh_official", False))
        or bool(getattr(args, "csv_geo_to_bvh_official_pos_only", False))
        or args.csv_apply_bvh_rotation
        or args.csv_units != "m"
    ):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from deploy_real.pose_csv_loader import (
            apply_bvh_like_coordinate_transform,
            convert_quat_order,
            default_parent_map_body,
            quats_local_to_global,
            gmr_rename_and_footmod,
            apply_geo_to_bvh_official,
            apply_geo_to_bvh_official_pos_only,
            load_csv_calib_json,
            apply_csv_calib_to_frames,
        )  # type: ignore

        # CSV calibration json (recommended for CSV-only replay)
        calib_path = str(getattr(args, "csv_calib_json", "")).strip()
        if calib_path:
            if not os.path.isabs(calib_path) and not os.path.exists(calib_path):
                cand = os.path.join(repo_root, calib_path)
                if os.path.exists(cand):
                    calib_path = cand
            if not os.path.exists(calib_path):
                print(f"❌ 找不到 csv_calib_json: {getattr(args, 'csv_calib_json', '')}", file=sys.stderr)
                return 2

            # Prevent double-transform footguns.
            if args.csv_apply_bvh_rotation or args.csv_geo_to_bvh_official or args.csv_geo_to_bvh_official_pos_only:
                print("❌ 使用 --csv_calib_json 时，请不要再叠加 --csv_apply_bvh_rotation/--csv_geo_to_bvh_official*（避免重复坐标变换）", file=sys.stderr)
                return 2
            if args.csv_units != "m":
                print("❌ 使用 --csv_calib_json 时，请不要再设置 --csv_units（单位由 calib.json 决定）", file=sys.stderr)
                return 2
            if args.csv_quat_order != "wxyz" or args.csv_quat_space != "global":
                print("❌ 使用 --csv_calib_json 时，请不要再设置 quat order/space（由 calib.json 决定）", file=sys.stderr)
                return 2

            calib = load_csv_calib_json(calib_path)
            frames = apply_csv_calib_to_frames(frames, calib, fmt=args.format, recompute_footmod=True)

        # quat order + local->global
        if args.csv_quat_order != "wxyz":
            frames = [convert_quat_order(fr, args.csv_quat_order) for fr in frames]
        if args.csv_quat_space == "local":
            pm = default_parent_map_body()
            frames = [quats_local_to_global(fr, pm) for fr in frames]

        # Vendor official Geo->BVH mapping (pos+quat or pos-only). Mutually exclusive with other axis mappings.
        if bool(getattr(args, "csv_geo_to_bvh_official", False)) or bool(getattr(args, "csv_geo_to_bvh_official_pos_only", False)):
            if bool(getattr(args, "csv_geo_to_bvh_official", False)) and bool(getattr(args, "csv_geo_to_bvh_official_pos_only", False)):
                print("⚠️ 同时设置了 --csv_geo_to_bvh_official 与 --csv_geo_to_bvh_official_pos_only，将优先使用 pos+quat 版本", file=sys.stderr)
            if bool(getattr(args, "csv_geo_to_bvh_official", False)):
                frames = [apply_geo_to_bvh_official(fr) for fr in frames]
            else:
                frames = [apply_geo_to_bvh_official_pos_only(fr) for fr in frames]

        # unit + optional BVH-like axis rotation
        if args.csv_apply_bvh_rotation or args.csv_units != "m":
            frames = [
                apply_bvh_like_coordinate_transform(fr, pos_unit=args.csv_units, apply_rotation=bool(args.csv_apply_bvh_rotation))
                for fr in frames
            ]
    n = len(frames)
    if n <= 0:
        print("❌ CSV 没有帧数据", file=sys.stderr)
        return 2

    start = max(0, int(args.start))
    end = n if int(args.end) < 0 else min(n, int(args.end))
    if start >= end:
        print(f"❌ start/end 非法: start={start}, end={end}, total={n}", file=sys.stderr)
        return 2

    # Init retargeter
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot="unitree_g1",
        actual_human_height=float(actual_human_height),
    )

    # Init redis
    client = redis.Redis(host=args.redis_ip, port=6379, db=0, decode_responses=False)
    try:
        client.ping()
    except Exception as e:
        print(f"❌ Redis 连接失败: {e}", file=sys.stderr)
        return 3

    # Keys (固定与 sim2real / recorder 对齐)
    robot_key = "unitree_g1_with_hands"
    key_action_body = f"action_body_{robot_key}"
    key_action_hand_l = f"action_hand_left_{robot_key}"
    key_action_hand_r = f"action_hand_right_{robot_key}"
    key_action_neck = f"action_neck_{robot_key}"
    key_t_action = "t_action"

    key_ht_l = f"hand_tracking_left_{robot_key}"
    key_ht_r = f"hand_tracking_right_{robot_key}"
    key_wuji_mode_l = f"wuji_hand_mode_left_{robot_key}"
    key_wuji_mode_r = f"wuji_hand_mode_right_{robot_key}"

    # Defaults (do not control hands/neck/wuji)
    default_hand_7 = np.zeros(7, dtype=float).tolist()
    default_neck_2 = [0.0, 0.0]

    # For mimic obs computation
    last_qpos: Optional[np.ndarray] = None
    fps = float(args.fps)
    if args.use_csv_fps and fps_from_csv and fps_from_csv > 1e-6:
        fps = float(fps_from_csv)
        print(f"[info] use_csv_fps=True, override fps => {fps}")
    dt = 1.0 / float(fps)
    rate = _RateLimiter(fps)

    print("=" * 70)
    print("3D Pose CSV Replay -> Redis (BODY ONLY)")
    print("=" * 70)
    print(f"csv_file: {args.csv_file}")
    print(f"format  : {args.format}")
    print(f"frames  : {n} (play [{start}, {end}))")
    print(f"fps     : {fps}")
    print(f"loop    : {args.loop}")
    print(f"redis   : {args.redis_ip}:6379")
    print(f"offset_to_ground: {args.offset_to_ground}")
    if args.dry_run:
        print("dry_run : True (不会写 Redis)")
    print("")

    # 仅打印一次：骨骼 key 样例，便于确认映射是否生效
    try:
        sample_keys = list(frames[start].keys())[:25]
        print(f"[hint] sample frame keys (first 25): {sample_keys}\n")
    except Exception:
        pass

    i = start
    step = 0
    while True:
        human_data = frames[i]
        qpos = retargeter.retarget(human_data, offset_to_ground=args.offset_to_ground)

        if last_qpos is None:
            # 第一帧速度不稳定，直接用“静态默认”更安全
            from data_utils.params import DEFAULT_MIMIC_OBS
            mimic = np.array(DEFAULT_MIMIC_OBS["unitree_g1"][:35], dtype=float)
        else:
            mimic = extract_mimic_obs_whole_body(qpos, last_qpos, dt=dt).astype(float)

        last_qpos = qpos.copy()

        if not args.dry_run:
            now_ms = _now_ms()
            pipe = client.pipeline()
            pipe.set(key_action_body, json.dumps(mimic.reshape(-1).tolist()))
            pipe.set(key_action_hand_l, json.dumps(default_hand_7))
            pipe.set(key_action_hand_r, json.dumps(default_hand_7))
            pipe.set(key_action_neck, json.dumps(default_neck_2))
            pipe.set(key_t_action, now_ms)

            # 强制 Wuji 不参与 replay：保持 default（回零位），并禁用 tracking
            pipe.set(key_wuji_mode_l, "default")
            pipe.set(key_wuji_mode_r, "default")
            pipe.set(key_ht_l, json.dumps({"is_active": False, "timestamp": now_ms}))
            pipe.set(key_ht_r, json.dumps({"is_active": False, "timestamp": now_ms}))
            pipe.execute()

        if args.print_every > 0 and (step % int(args.print_every) == 0):
            print(f"[replay] frame={i} step={step} t_action_ms={_now_ms()}")

        # advance frame
        i += 1
        step += 1
        if i >= end:
            if args.loop:
                i = start
            else:
                break

        rate.sleep()

    print("✅ replay finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


