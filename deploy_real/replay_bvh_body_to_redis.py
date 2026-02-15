#!/usr/bin/env python3
"""
Replay BVH motion to G1 via Redis (BODY ONLY).

目标：
- 读取 BVH（lafan1 / nokov）并通过 GMR 做全身重定向
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
        # Normalize common BVH naming (case/underscore) -> canonical names
        def _norm(s: str) -> str:
            return "".join([c.lower() for c in str(s) if c.isalnum()])
        norm_map = {
            "hip": "Hips",
            "hips": "Hips",
            "waist": "Spine",
            "spine": "Spine",
            "spine1": "Spine1",
            "spine2": "Spine2",
            "spine3": "Spine3",
            "chest": "Spine1",
            "upperchest": "Spine2",
            "neck": "Neck",
            "head": "Head",
            "leftupperleg": "LeftUpLeg",
            "rightupperleg": "RightUpLeg",
            "leftlowerleg": "LeftLeg",
            "rightlowerleg": "RightLeg",
            "leftupperarm": "LeftArm",
            "rightupperarm": "RightArm",
            "leftlowerarm": "LeftForeArm",
            "rightlowerarm": "RightForeArm",
            "leftfoot": "LeftFoot",
            "rightfoot": "RightFoot",
            "lefthand": "LeftHand",
            "righthand": "RightHand",
            "lefttoe": "LeftToe",
            "righttoe": "RightToe",
            "lefttoebase": "LeftToeBase",
            "righttoebase": "RightToeBase",
            "leftfoottracker": "LeftFoot",
            "rightfoottracker": "RightFoot",
            "lefttoetracker": "LeftToe",
            "righttoetracker": "RightToe",
        }
        for k, v in list(out.items()):
            nk = _norm(k)
            if nk in norm_map:
                dst = norm_map[nk]
                if dst not in out:
                    out[dst] = v
        # Toe / ToeBase 双向别名（便于 nokov/lafan1 两种 config）
        if "LeftToe" in out and "LeftToeBase" not in out:
            out["LeftToeBase"] = out["LeftToe"]
        if "RightToe" in out and "RightToeBase" not in out:
            out["RightToeBase"] = out["RightToe"]
        if "LeftToeBase" in out and "LeftToe" not in out:
            out["LeftToe"] = out["LeftToeBase"]
        if "RightToeBase" in out and "RightToe" not in out:
            out["RightToe"] = out["RightToeBase"]
        # Fallback: if still missing Toe, use Foot orientation
        if "LeftToe" not in out and "LeftFoot" in out:
            out["LeftToe"] = out["LeftFoot"]
        if "RightToe" not in out and "RightFoot" in out:
            out["RightToe"] = out["RightFoot"]
        # Fallback spine chain if missing
        if "Spine" not in out and "Hips" in out:
            out["Spine"] = out["Hips"]
        if "Spine1" not in out:
            if "Spine" in out:
                out["Spine1"] = out["Spine"]
            elif "Chest" in out:
                out["Spine1"] = out["Chest"]
        if "Spine2" not in out:
            if "Spine1" in out:
                out["Spine2"] = out["Spine1"]
            elif "UpperChest" in out:
                out["Spine2"] = out["UpperChest"]
        if "Spine3" not in out:
            if "Spine2" in out:
                out["Spine3"] = out["Spine2"]
            elif "Neck" in out:
                out["Spine3"] = out["Neck"]
        if "Neck" not in out and "Spine3" in out:
            out["Neck"] = out["Spine3"]
        if "Head" not in out and "Neck" in out:
            out["Head"] = out["Neck"]
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


def _load_bvh_frames_fk_raw_world(bvh_file: str, fmt: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    Load BVH and compute FK global pos/quat in the BVH *raw world* (no fixed axis rotation).

    - Still converts position from cm -> m (so it's comparable to CSV meters)
    - Still applies the same rename aliases and FootMod construction as _load_bvh_frames_via_gmr
    - Orientation is the FK global quaternion directly (scalar-first / wxyz in our pipeline usage)

    This is useful when you want to first align CSV->BVH world, and only then apply the
    fixed BVH->GMR world rotation as a separate step.
    """
    import general_motion_retargeting.utils.lafan_vendor.utils as utils  # type: ignore
    from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh  # type: ignore

    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    def _rename(result: Dict[str, Any]) -> Dict[str, Any]:
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
            orientation = global_data[0][frame, i]
            position = global_data[1][frame, i] / 100.0  # cm -> m (raw BVH world)
            result[bone] = [position, orientation]

        result = _rename(result)

        # FootMod selection same as main loader
        if fmt == "lafan1":
            left_toe = "LeftToe" if "LeftToe" in result else "LeftToeBase"
            right_toe = "RightToe" if "RightToe" in result else "RightToeBase"
            result["LeftFootMod"] = [result["LeftFoot"][0], result[left_toe][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result[right_toe][1]]
        elif fmt == "nokov":
            left_toe = "LeftToeBase" if "LeftToeBase" in result else "LeftToe"
            right_toe = "RightToeBase" if "RightToeBase" in result else "RightToe"
            result["LeftFootMod"] = [result["LeftFoot"][0], result[left_toe][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result[right_toe][1]]
        else:
            raise ValueError(f"Invalid format: {fmt}")

        frames.append(result)

    human_height = 1.75
    return frames, human_height


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
    parser = argparse.ArgumentParser(description="Replay BVH to G1 via Redis (body only)")
    parser.add_argument("--bvh_file", type=str, required=True, help="Path to BVH file")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1", help="BVH format")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis IP")
    parser.add_argument("--fps", type=float, default=30.0, help="Replay FPS (default: 30)")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--start", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (exclusive), -1 for end")
    parser.add_argument("--offset_to_ground", action="store_true", help="Offset human motion to ground")
    parser.add_argument("--print_every", type=int, default=60, help="Print status every N frames")
    parser.add_argument("--dry_run", action="store_true", help="Do not write Redis, only run retarget")
    args = parser.parse_args()

    GMR, load_bvh_file = _import_gmr()

    # Resolve BVH path: replay_bvh.sh 会 cd 到 deploy_real，用户常给的是仓库根目录相对路径
    bvh_path = args.bvh_file
    if not os.path.isabs(bvh_path) and not os.path.exists(bvh_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cand = os.path.join(repo_root, bvh_path)
        if os.path.exists(cand):
            bvh_path = cand
    if not os.path.exists(bvh_path):
        print(f"❌ 找不到 BVH 文件: {args.bvh_file}", file=sys.stderr)
        print("   你可以传绝对路径，或在仓库根目录运行脚本。", file=sys.stderr)
        return 2

    # Load BVH frames (human data dict list)
    frames, actual_human_height = load_bvh_file(bvh_path, args.format)
    n = len(frames)
    if n <= 0:
        print("❌ BVH 没有帧数据", file=sys.stderr)
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
    dt = 1.0 / float(args.fps)
    rate = _RateLimiter(args.fps)

    print("=" * 70)
    print("BVH Replay -> Redis (BODY ONLY)")
    print("=" * 70)
    print(f"bvh_file: {args.bvh_file}")
    print(f"format  : {args.format}")
    print(f"frames  : {n} (play [{start}, {end}))")
    print(f"fps     : {args.fps}")
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


