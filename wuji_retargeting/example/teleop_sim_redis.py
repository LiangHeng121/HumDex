"""
Wuji Hand Controller via Redis (Policy Inference) - SIMULATION

目标：与 `wuji_retarget/deploy2.py` 的数据流保持一致，只有最后的执行端不同：
- real：下发到 wujihandpy 硬件
- sim ：下发到 MuJoCo 的 data.ctrl
"""

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

try:
    import redis  # type: ignore
except Exception:
    redis = None

TWIST2_ROOT = Path(__file__).resolve().parents[2]
WUJI_RETARGET_DIR = TWIST2_ROOT / "wuji_retarget"
WUJI_RETARGETING_DIR = TWIST2_ROOT / "wuji_retargeting"

# 对齐 deploy2.py：确保能 import 到仓库内的本地模块（geort / wuji_retargeting）
for p in [str(WUJI_RETARGET_DIR), str(WUJI_RETARGETING_DIR), str(TWIST2_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import geort
from wuji_retargeting.mediapipe import apply_mediapipe_transformations


# ------------------------------
# 26D joint names (upstream format) (与 deploy2.py 保持一致)
# ------------------------------
HAND_JOINT_NAMES_26 = [
    "Wrist",
    "Palm",
    "ThumbMetacarpal",
    "ThumbProximal",
    "ThumbDistal",
    "ThumbTip",
    "IndexMetacarpal",
    "IndexProximal",
    "IndexIntermediate",
    "IndexDistal",
    "IndexTip",
    "MiddleMetacarpal",
    "MiddleProximal",
    "MiddleIntermediate",
    "MiddleDistal",
    "MiddleTip",
    "RingMetacarpal",
    "RingProximal",
    "RingIntermediate",
    "RingDistal",
    "RingTip",
    "LittleMetacarpal",
    "LittleProximal",
    "LittleIntermediate",
    "LittleDistal",
    "LittleTip",
]

# ------------------------------
# 26D -> 21D mapping (与 deploy2.py 保持一致)
# 注意：0 点来自 index=1（Palm）
# ------------------------------
MEDIAPIPE_MAPPING_26_TO_21 = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    20,
    21,
    22,
    23,
    25,
]


def now_ms() -> int:
    return int(time.time() * 1000)


def _decode_redis_value(val) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore")
    return str(val)


def _redis_cli_ping(host: str, port: int) -> bool:
    redis_cli = shutil.which("redis-cli")
    if not redis_cli:
        return False
    try:
        out = subprocess.check_output([redis_cli, "-h", host, "-p", str(port), "PING"], stderr=subprocess.DEVNULL)
        return out.strip().upper() == b"PONG"
    except Exception:
        return False


def _redis_cli_get(host: str, port: int, key: str) -> Optional[bytes]:
    redis_cli = shutil.which("redis-cli")
    if not redis_cli:
        return None
    try:
        out = subprocess.check_output([redis_cli, "-h", host, "-p", str(port), "GET", key], stderr=subprocess.DEVNULL)
        if out.strip() == b"(nil)":
            return None
        return out
    except Exception:
        return None


def _redis_cli_set(host: str, port: int, key: str, value: str) -> bool:
    redis_cli = shutil.which("redis-cli")
    if not redis_cli:
        return False
    try:
        out = subprocess.check_output(
            [redis_cli, "-h", host, "-p", str(port), "SET", key, value],
            stderr=subprocess.DEVNULL,
        )
        return out.strip().upper() == b"OK"
    except Exception:
        return False


@dataclass
class _RedisIO:
    host: str
    port: int
    client: Optional[Any]

    def get(self, key: str) -> Optional[str]:
        if self.client is not None:
            v = self.client.get(key)
            return _decode_redis_value(v)
        v = _redis_cli_get(self.host, self.port, key)
        return _decode_redis_value(v)

    def set(self, key: str, value: str) -> bool:
        if self.client is not None:
            try:
                self.client.set(key, value)
                return True
            except Exception:
                return False
        return _redis_cli_set(self.host, self.port, key, value)


def hand_26d_to_mediapipe_21d(hand_data_dict, hand_side="left") -> np.ndarray:
    """与 deploy2.py 同源：读取 26 个关节 xyz，按 mapping 取出 21 点，并以 wrist 为原点。"""
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"
    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)

    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_data_dict:
            pos = hand_data_dict[key][0]  # [x,y,z]
            joint_positions_26[i] = pos
        else:
            joint_positions_26[i] = [0.0, 0.0, 0.0]

    mediapipe_21d = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21]

    wrist_pos = mediapipe_21d[0].copy()
    mediapipe_21d = mediapipe_21d - wrist_pos
    return mediapipe_21d


def _apply_safety(qpos_5x4: np.ndarray, last_qpos: np.ndarray, clamp_min: float, clamp_max: float, max_delta: float) -> np.ndarray:
    q = np.asarray(qpos_5x4, dtype=np.float32).reshape(5, 4)
    q = np.clip(q, clamp_min, clamp_max)
    if last_qpos is not None and last_qpos.shape == q.shape:
        delta = q - last_qpos
        delta = np.clip(delta, -max_delta, max_delta)
        q = last_qpos + delta
    return q


def smooth_move_sim(data: mujoco.MjData, target_qpos_5x4: np.ndarray, duration: float, steps: int):
    target = np.asarray(target_qpos_5x4, dtype=np.float32).reshape(-1)
    cur = np.asarray(data.ctrl, dtype=np.float32).reshape(-1)
    n = min(cur.shape[0], target.shape[0])
    for t in np.linspace(0.0, 1.0, max(int(steps), 1)):
        q = cur.copy()
        q[:n] = cur[:n] * (1 - t) + target[:n] * t
        data.ctrl[:n] = q[:n]
        time.sleep(max(duration, 0.0) / max(int(steps), 1))


def get_hand_tracking_data_from_redis(rio: _RedisIO, key: str, stale_ms: int) -> Tuple[Optional[bool], Optional[dict]]:
    try:
        s = rio.get(key)
        if not s:
            return None, None
        hand_data = json.loads(s)
        if not isinstance(hand_data, dict):
            return None, None

        data_timestamp = int(hand_data.get("timestamp", 0) or 0)
        if data_timestamp > 0:
            if now_ms() - data_timestamp > int(stale_ms):
                return None, None

        is_active = bool(hand_data.get("is_active", False))
        if not is_active:
            return None, None

        hand_dict = {k: v for k, v in hand_data.items() if k not in ["is_active", "timestamp"]}
        return is_active, hand_dict
    except Exception:
        return None, None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Redis teleop -> 26D -> 21D -> policy inference -> MuJoCo sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hand_side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--redis_ip", type=str, default="localhost")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument("--target_fps", type=int, default=50)
    parser.add_argument("--stale_ms", type=int, default=500)
    parser.add_argument("--robot_key", type=str, default="unitree_g1_with_hands")

    parser.add_argument("--no_smooth", action="store_true")
    parser.add_argument("--smooth_steps", type=int, default=5)

    parser.add_argument("--policy_tag", type=str, default="geort_filter_wuji")
    parser.add_argument("--policy_epoch", type=int, default=-1)
    parser.add_argument("--use_fingertips5", action="store_true")
    parser.set_defaults(use_fingertips5=True)

    parser.add_argument("--clamp_min", type=float, default=-1.5)
    parser.add_argument("--clamp_max", type=float, default=1.5)
    parser.add_argument("--max_delta_per_step", type=float, default=0.08)
    return parser.parse_args()


def main():
    args = parse_arguments()

    hand_side = args.hand_side.lower()
    robot_key = str(args.robot_key)

    # redis keys (与 deploy2.py 一致)
    redis_key_hand_tracking = f"hand_tracking_{hand_side}_{robot_key}"
    redis_key_action_wuji_qpos_target = f"action_wuji_qpos_target_{hand_side}_{robot_key}"
    redis_key_state_wuji_hand = f"state_wuji_hand_{hand_side}_{robot_key}"
    redis_key_t_action_wuji_hand = f"t_action_wuji_hand_{hand_side}_{robot_key}"
    redis_key_t_state_wuji_hand = f"t_state_wuji_hand_{hand_side}_{robot_key}"
    redis_key_wuji_mode = f"wuji_hand_mode_{hand_side}_{robot_key}"

    # connect redis (python redis 优先，没装则 fallback redis-cli)
    client = None
    if redis is not None:
        try:
            client = redis.Redis(host=args.redis_ip, port=args.redis_port, decode_responses=False)
            client.ping()
        except Exception:
            client = None
    if client is None:
        if not _redis_cli_ping(args.redis_ip, args.redis_port):
            raise RuntimeError("Redis 不可用：python redis 不可用且 redis-cli PING 失败。")
        print("[info] 使用 redis-cli 模式（未安装 python redis 或连接失败）。")

    rio = _RedisIO(host=args.redis_ip, port=args.redis_port, client=client)

    # init policy (与 deploy2.py 一致)
    print("🔄 加载 Retarget Policy Model...")
    policy = geort.load_model(str(args.policy_tag), epoch=int(args.policy_epoch))
    try:
        policy.eval()
    except Exception:
        pass
    print(f"✅ Policy loaded: tag={args.policy_tag}, epoch={args.policy_epoch}")

    # Load MuJoCo model (仿真差异点)
    mujoco_sim_path = Path(__file__).parent / "utils" / "mujoco-sim"
    mjcf_path = mujoco_sim_path / "model" / f"{hand_side}.xml"
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {mjcf_path}")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    # zero pose（与 deploy2.py 语义一致：全 0）
    zero_pose = np.zeros((5, 4), dtype=np.float32)
    last_qpos = zero_pose.copy()

    # stabilize
    for _ in range(100):
        mujoco.mj_step(model, data)

    target_fps = int(args.target_fps)
    control_dt = 1.0 / float(max(target_fps, 1))

    running = True

    def _handle_signal(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("=" * 60)
    print("Wuji Hand Controller via Redis (SIM)")
    print("=" * 60)
    print(f"手部: {hand_side}")
    print(f"Redis: {args.redis_ip}:{args.redis_port}")
    print(f"Keys: {redis_key_hand_tracking} / {redis_key_action_wuji_qpos_target} / {redis_key_wuji_mode}")
    print(f"目标频率: {target_fps} Hz")
    print(f"平滑移动: {'禁用' if args.no_smooth else '启用'} steps={args.smooth_steps}")
    print(f"Safety: clamp=[{args.clamp_min},{args.clamp_max}], max_delta={args.max_delta_per_step}")
    print("=" * 60)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.5
        viewer.cam.lookat[:] = [0, 0, 0.05]

        while running and viewer.is_running():
            loop_start = time.time()

            # 0) read mode (与 deploy2.py 一致)
            mode = (rio.get(redis_key_wuji_mode) or "follow").strip().lower()

            if mode in ["default", "hold"]:
                target = zero_pose if mode == "default" else last_qpos

                # write action target
                rio.set(redis_key_action_wuji_qpos_target, json.dumps(target.reshape(-1).tolist()))
                rio.set(redis_key_t_action_wuji_hand, str(now_ms()))

                # control sim
                if args.no_smooth:
                    data.ctrl[: min(model.nu, 20)] = target.reshape(-1)[: min(model.nu, 20)]
                else:
                    smooth_move_sim(data, target, duration=control_dt, steps=int(args.smooth_steps))

                # write state
                rio.set(redis_key_state_wuji_hand, json.dumps(np.asarray(data.ctrl[: min(model.nu, 20)]).reshape(-1).tolist()))
                rio.set(redis_key_t_state_wuji_hand, str(now_ms()))

                mujoco.mj_step(model, data)
                viewer.sync()
            else:
                # follow: read tracking -> 26D->21D->transform->policy->safety
                is_active, hand_data_dict = get_hand_tracking_data_from_redis(
                    rio, redis_key_hand_tracking, stale_ms=int(args.stale_ms)
                )

                if is_active and hand_data_dict is not None:
                    mediapipe_21d = hand_26d_to_mediapipe_21d(hand_data_dict, hand_side=hand_side)
                    mediapipe_transformed = apply_mediapipe_transformations(mediapipe_21d, hand_type=hand_side)

                    pts21 = np.asarray(mediapipe_transformed, dtype=np.float32).reshape(21, 3)
                    if bool(args.use_fingertips5):
                        human_points = pts21[[4, 8, 12, 16, 20], :3]  # (5,3)
                    else:
                        human_points = pts21

                    if _TORCH_AVAILABLE and torch is not None:
                        with torch.no_grad():
                            qpos_20 = policy.forward(human_points)
                    else:
                        qpos_20 = policy.forward(human_points)

                    qpos_20 = np.asarray(qpos_20, dtype=np.float32).reshape(-1)
                    if qpos_20.shape[0] != 20:
                        raise ValueError(f"Policy output dim mismatch: expect 20, got {qpos_20.shape[0]}")

                    wuji_20d = qpos_20.reshape(5, 4)
                    wuji_20d = _apply_safety(
                        wuji_20d,
                        last_qpos=last_qpos,
                        clamp_min=float(args.clamp_min),
                        clamp_max=float(args.clamp_max),
                        max_delta=float(args.max_delta_per_step),
                    )

                    # write action target
                    rio.set(redis_key_action_wuji_qpos_target, json.dumps(wuji_20d.reshape(-1).tolist()))
                    rio.set(redis_key_t_action_wuji_hand, str(now_ms()))

                    # control sim
                    if args.no_smooth:
                        data.ctrl[: min(model.nu, 20)] = wuji_20d.reshape(-1)[: min(model.nu, 20)]
                    else:
                        smooth_move_sim(data, wuji_20d, duration=control_dt, steps=int(args.smooth_steps))

                    # write state (sim: ctrl 作为状态回写)
                    rio.set(redis_key_state_wuji_hand, json.dumps(np.asarray(data.ctrl[: min(model.nu, 20)]).reshape(-1).tolist()))
                    rio.set(redis_key_t_state_wuji_hand, str(now_ms()))

                    last_qpos = wuji_20d.copy()

                mujoco.mj_step(model, data)
                viewer.sync()

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, control_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()


