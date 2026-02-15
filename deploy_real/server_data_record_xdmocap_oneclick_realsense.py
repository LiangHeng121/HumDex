#!/usr/bin/env python3
"""
纯录制/数据处理脚本（不启动机器人/手控制）：

- 可选：后台启动 XDMocap -> Redis（复用 `xdmocap_teleop.sh`）
- 主进程只负责：
  - 本机直连 RealSense 采图
  - 从 Redis 读取 state/action/hand_tracking 等 key
  - 写入 EpisodeWriter

键盘：
- r: 开始/停止录制
- q: 退出

安全说明：
- 为避免误操作，本脚本 **不会** 启动 sim2real/Wuji 控制；即便传入 `--start_sim2real/--start_wuji`
  也会忽略并提示。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import redis
from rich import print

# Ensure repo root is on sys.path so `import deploy_real.*` works even when
# launching from inside `deploy_real/` (e.g. via `bash data_record_xdmocap_oneclick.sh`).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy_real.data_utils.episode_writer import EpisodeWriter


def now_ms() -> int:
    return int(time.time() * 1000)


def safe_json_loads(raw: Optional[bytes]) -> Any:
    if raw is None:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None


# -----------------------------
# Optional: local Wuji retarget (no wuji_server required)
# -----------------------------

# 26维手部关节名称（与 teleop 写入 hand_tracking_* 的 key 一致）
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

# 26维到21维 MediaPipe 格式的映射索引
# MediaPipe 格式: [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
# 26维格式: [Wrist, Palm, Thumb(4), Index(5), Middle(5), Ring(5), Pinky(5)]
MEDIAPIPE_MAPPING_26_TO_21 = [
    1,  # 0: Wrist -> Wrist (这里使用 Palm 作为 Wrist，与 server_wuji_hand_redis.py 保持一致)
    2,  # 1: ThumbMetacarpal -> Thumb CMC
    3,  # 2: ThumbProximal -> Thumb MCP
    4,  # 3: ThumbDistal -> Thumb IP
    5,  # 4: ThumbTip -> Thumb Tip
    6,  # 5: IndexMetacarpal -> Index MCP
    7,  # 6: IndexProximal -> Index PIP
    8,  # 7: IndexIntermediate -> Index DIP
    10,  # 8: IndexTip -> Index Tip (跳过 IndexDistal)
    11,  # 9: MiddleMetacarpal -> Middle MCP
    12,  # 10: MiddleProximal -> Middle PIP
    13,  # 11: MiddleIntermediate -> Middle DIP
    15,  # 12: MiddleTip -> Middle Tip (跳过 MiddleDistal)
    16,  # 13: RingMetacarpal -> Ring MCP
    17,  # 14: RingProximal -> Ring PIP
    18,  # 15: RingIntermediate -> Ring DIP
    20,  # 16: RingTip -> Ring Tip (跳过 RingDistal)
    21,  # 17: LittleMetacarpal -> Pinky MCP
    22,  # 18: LittleProximal -> Pinky PIP
    23,  # 19: LittleIntermediate -> Pinky DIP
    25,  # 20: LittleTip -> Pinky Tip (跳过 LittleDistal)
]


def hand_26d_to_mediapipe_21d(hand_data_dict: Dict[str, Any], hand_side: str = "left") -> np.ndarray:
    """
    将26维 hand_tracking dict 转换为21维 MediaPipe 格式 (21,3)。
    注意：实现与 `deploy_real/server_wuji_hand_redis.py::hand_26d_to_mediapipe_21d` 对齐。
    """
    side = str(hand_side).lower()
    prefix = "LeftHand" if side == "left" else "RightHand"
    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = prefix + joint_name
        v = hand_data_dict.get(key, None)
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            pos = np.asarray(v[0], dtype=np.float32).reshape(3)
            joint_positions_26[i] = pos
        else:
            joint_positions_26[i] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    mp21 = joint_positions_26[np.asarray(MEDIAPIPE_MAPPING_26_TO_21, dtype=np.int32)]
    wrist_pos = mp21[0].copy()
    mp21 = mp21 - wrist_pos
    # scale_factor=1.0 保持一致（这里不做额外放缩）
    return mp21


def _try_import_wuji_retargeting():
    """
    Lazy import to avoid hard dependency when user doesn't need local retarget.
    Returns (WujiHandRetargeter, apply_mediapipe_transformations) or (None, None)
    """
    try:
        project_root = Path(__file__).resolve().parents[1]
        wuji_path = project_root / "wuji_retargeting"
        if str(wuji_path) not in sys.path:
            sys.path.insert(0, str(wuji_path))
        from wuji_retargeting import WujiHandRetargeter  # type: ignore
        from wuji_retargeting.mediapipe import apply_mediapipe_transformations  # type: ignore

        return WujiHandRetargeter, apply_mediapipe_transformations
    except Exception:
        return None, None


def _try_import_geort():
    """
    Lazy import GeoRT model package from repo's `wuji_retarget/`.
    Returns geort module or None.
    """
    try:
        project_root = Path(__file__).resolve().parents[1]
        geort_root = project_root / "wuji_retarget"
        if str(geort_root) not in sys.path:
            sys.path.insert(0, str(geort_root))
        import geort  # type: ignore

        return geort
    except Exception:
        return None


class RealSenseVisionSource:
    """Direct RealSense capture (color; optional depth) via pyrealsense2."""

    def __init__(self, width: int, height: int, fps: int, enable_depth: bool = False):
        try:
            import pyrealsense2 as rs  # type: ignore
        except Exception as e:
            raise ImportError("未安装 pyrealsense2，无法使用 --vision_backend realsense。") from e

        self._rs = rs
        self._enable_depth = bool(enable_depth)
        self._running = True
        self._lock = None  # lazy: avoid threading unless needed

        self._latest_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self._latest_depth: Optional[np.ndarray] = None

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if self._enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(cfg)

    def get_rgb(self) -> np.ndarray:
        rs = self._rs
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color = frames.get_color_frame()
            if color:
                self._latest_rgb = np.asanyarray(color.get_data())
            if self._enable_depth:
                depth = frames.get_depth_frame()
                self._latest_depth = None if not depth else np.asanyarray(depth.get_data())
        except Exception:
            pass
        return self._latest_rgb.copy()

    def get_depth(self) -> Optional[np.ndarray]:
        return None if self._latest_depth is None else self._latest_depth.copy()

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")
    here = os.path.dirname(os.path.abspath(__file__))
    default_data_folder = os.path.join(here, "twist2_demonstration")

    p = argparse.ArgumentParser(description="XDMocap/Redis + RealSense 录制器（record-only；不启动机器人/手控制）")

    # recording
    p.add_argument("--data_folder", default=default_data_folder)
    p.add_argument("--task_name", default=cur_time)
    p.add_argument("--frequency", type=int, default=30)
    p.add_argument("--robot_key", default="unitree_g1_with_hands")
    p.add_argument("--no_window", action="store_true", help="不显示窗口（无窗口将无法按键控制录制）")
    p.add_argument("--record_on_start", action="store_true", help="启动后立刻开始录制（适用于 --no_window）")

    # redis
    p.add_argument("--redis_ip", default="localhost")
    p.add_argument("--redis_port", type=int, default=6379)

    # local wuji retarget (record-only; does NOT control hardware)
    p.add_argument("--local_wuji_retarget", type=int, default=1, help="是否在录制端本地计算 wuji qpos_target（0/1，默认1）")
    p.add_argument(
        "--local_wuji_retarget_overwrite",
        type=int,
        default=1,
        help="若 Redis 已有 action_wuji_qpos_target_*，是否用本地 retarget 覆盖（0/1，默认0）",
    )
    p.add_argument("--local_wuji_write_redis", type=int, default=1, help="把本地 retarget 结果写回 Redis（0/1，默认0）")

    # local wuji mode: DexPilot retarget (default) vs GeoRT model inference
    p.add_argument("--local_wuji_use_model", type=int, default=0, help="本地 wuji 目标是否使用 GeoRT 模型推理（0/1，默认0=原 retarget）")
    # 参数名对齐 deploy2.py / wuji_hand_model_deploy.sh（policy_tag/policy_epoch）
    p.add_argument("--local_wuji_policy_tag", type=str, default="geort_filter_wuji", help="本地 GeoRT 模型 tag（--local_wuji_use_model=1）")
    p.add_argument("--local_wuji_policy_epoch", type=int, default=-1, help="本地 GeoRT 模型 epoch（--local_wuji_use_model=1）")
    p.add_argument("--local_wuji_policy_tag_left", type=str, default="", help="左手 tag（可选；空则用 local_wuji_policy_tag）")
    p.add_argument("--local_wuji_policy_epoch_left", type=int, default=-999999, help="左手 epoch（可选；-999999 表示用 local_wuji_policy_epoch）")
    p.add_argument("--local_wuji_policy_tag_right", type=str, default="", help="右手 tag（可选；空则用 local_wuji_policy_tag）")
    p.add_argument("--local_wuji_policy_epoch_right", type=int, default=-999999, help="右手 epoch（可选；-999999 表示用 local_wuji_policy_epoch）")
    p.add_argument("--local_wuji_use_fingertips5", type=int, default=1, help="model 输入用 5 指尖 (5,3)（0/1，默认1）")
    # safety（主要给 model 模式，避免录制到异常抖动/越界）
    p.add_argument("--local_wuji_clamp_min", type=float, default=-1.5, help="model 输出限幅最小值")
    p.add_argument("--local_wuji_clamp_max", type=float, default=1.5, help="model 输出限幅最大值")
    p.add_argument("--local_wuji_max_delta_per_step", type=float, default=0.08, help="model 输出每步最大变化")

    # vision (realsense only per requirement)
    p.add_argument("--rs_w", type=int, default=640)
    p.add_argument("--rs_h", type=int, default=480)
    p.add_argument("--rs_fps", type=int, default=30)
    p.add_argument("--rs_depth", action="store_true")

    # xdmocap teleop (as subprocess)
    p.add_argument("--start_xdmocap", action="store_true", help="后台启动 xdmocap_teleop.sh")
    p.add_argument("--xdmocap_teleop_sh", type=str, default=os.path.join(os.path.dirname(here), "xdmocap_teleop.sh"))
    p.add_argument("--xdmocap_extra_args", nargs=argparse.REMAINDER, default=[], help="追加给 xdmocap_teleop.sh 的参数（放在 -- 之后）")

    # 兼容旧参数：保留但本脚本不会启动机器人/手控制
    p.add_argument("--start_sim2real", action="store_true", help="(deprecated/ignored) 仅保留兼容性，本脚本不会启动机器人控制")
    p.add_argument("--policy", type=str, default=os.path.join(os.path.dirname(here), "assets/ckpts/twist2_1017_20k.onnx"))
    p.add_argument("--config", type=str, default="robot_control/configs/g1.yaml")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--net", type=str, default="eno1")
    p.add_argument("--use_hand", action="store_true", help="是否启用 Unitree 夹爪手（Dex3_1_Controller）")
    p.add_argument("--record_proprio", action="store_true")
    p.add_argument("--smooth_body", type=float, default=0.0)
    p.add_argument("--safety_rate_limit", action="store_true")
    p.add_argument("--safety_rate_limit_scope", choices=["all", "arms"], default="arms")
    p.add_argument("--max_dof_delta_per_step", type=float, default=1.0)
    p.add_argument("--max_dof_delta_print_every", type=int, default=200)

    # 兼容旧参数：保留但本脚本不会启动机器人/手控制
    p.add_argument("--start_wuji", action="store_true", help="(deprecated/ignored) 仅保留兼容性，本脚本不会启动 Wuji 手控制")
    p.add_argument("--wuji_hands", choices=["none", "left", "right", "both"], default="right")
    p.add_argument("--wuji_target_fps", type=int, default=50)
    p.add_argument("--wuji_no_smooth", action="store_true")
    p.add_argument("--wuji_smooth_steps", type=int, default=5)
    p.add_argument("--wuji_left_serial", type=str, default="")
    p.add_argument("--wuji_right_serial", type=str, default="")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Redis connection (main recorder)
    try:
        pool = redis.ConnectionPool(
            host=args.redis_ip,
            port=args.redis_port,
            db=0,
            max_connections=10,
            retry_on_timeout=True,
            socket_timeout=0.2,
            socket_connect_timeout=0.2,
        )
        client = redis.Redis(connection_pool=pool)
        pipe = client.pipeline()
        client.ping()
    except Exception as e:
        print(f"❌ Redis 连接失败: {e}")
        return 2

    # Start xdmocap teleop as background subprocess
    xdm_proc: Optional[subprocess.Popen] = None
    if bool(args.start_xdmocap):
        teleop_sh = os.path.abspath(str(args.xdmocap_teleop_sh))
        if not os.path.exists(teleop_sh):
            print(f"❌ 找不到 xdmocap_teleop.sh: {teleop_sh}")
            return 2
        cmd = ["bash", teleop_sh] + list(args.xdmocap_extra_args or [])
        print(f"[xdmocap] starting: {' '.join(cmd)}")
        xdm_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)

    # Safety: explicitly ignore controller-start flags (compatibility only)
    if bool(args.start_sim2real):
        print("[WARN] --start_sim2real 已被忽略：本脚本是纯录制器，不会启动机器人控制。")
    if bool(args.start_wuji):
        print("[WARN] --start_wuji 已被忽略：本脚本是纯录制器，不会启动 Wuji 手控制。")

    # Vision (RealSense direct)
    try:
        vision = RealSenseVisionSource(width=int(args.rs_w), height=int(args.rs_h), fps=int(args.rs_fps), enable_depth=bool(args.rs_depth))
    except Exception as e:
        print(f"❌ RealSense 初始化失败: {e}")
        if xdm_proc is not None:
            try:
                os.killpg(os.getpgid(xdm_proc.pid), signal.SIGTERM)
            except Exception:
                pass
        return 3

    # Recorder (schema: compatible superset of data_record.sh + keyboard recorder)
    task_dir = os.path.join(str(args.data_folder), str(args.task_name))
    recorder = EpisodeWriter(
        task_dir=task_dir,
        frequency=int(args.frequency),
        image_shape=(int(args.rs_h), int(args.rs_w), 3),
        data_keys=["rgb"],
    )

    suffix = str(args.robot_key)
    redis_keys: List[str] = [
        f"state_body_{suffix}",
        f"state_hand_left_{suffix}",
        f"state_hand_right_{suffix}",
        f"state_neck_{suffix}",
        "t_state",
        f"action_body_{suffix}",
        f"action_hand_left_{suffix}",
        f"action_hand_right_{suffix}",
        f"action_neck_{suffix}",
        "t_action",
        # teleop -> wuji inputs
        f"hand_tracking_left_{suffix}",
        f"hand_tracking_right_{suffix}",
        # wuji outputs (optional)
        f"action_wuji_qpos_target_left_{suffix}",
        f"action_wuji_qpos_target_right_{suffix}",
        f"t_action_wuji_hand_left_{suffix}",
        f"t_action_wuji_hand_right_{suffix}",
        f"state_wuji_hand_left_{suffix}",
        f"state_wuji_hand_right_{suffix}",
        f"t_state_wuji_hand_left_{suffix}",
        f"t_state_wuji_hand_right_{suffix}",
    ]
    data_dict_keys: List[str] = [
        "state_body",
        "state_hand_left",
        "state_hand_right",
        "state_neck",
        "t_state",
        "action_body",
        "action_hand_left",
        "action_hand_right",
        "action_neck",
        "t_action",
        "hand_tracking_left",
        "hand_tracking_right",
        "action_wuji_qpos_target_left",
        "action_wuji_qpos_target_right",
        "t_action_wuji_hand_left",
        "t_action_wuji_hand_right",
        "state_wuji_hand_left",
        "state_wuji_hand_right",
        "t_state_wuji_hand_left",
        "t_state_wuji_hand_right",
    ]

    window_name = "TWIST2 OneClick Recorder (r=start/stop, q=quit)"
    if not args.no_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    recording = False
    step_count = 0
    control_dt = 1.0 / max(1.0, float(args.frequency))

    stop_requested = False

    def _handle_sig(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    print("=" * 70)
    print("XDMocap/Redis + RealSense Recorder (record-only)")
    print("=" * 70)
    print(f"- save_to: {task_dir}")
    print(f"- redis: {args.redis_ip}:{args.redis_port}  suffix={suffix}")
    print(f"- rs: {args.rs_w}x{args.rs_h}@{args.rs_fps} depth={bool(args.rs_depth)}")
    print(f"- start_xdmocap: {bool(args.start_xdmocap)}")
    print(f"- record_on_start: {bool(args.record_on_start)}")
    print("Keys: r=start/stop, q=quit (注意：--no_window 时无法按键)")
    print("=" * 70)

    if bool(args.record_on_start):
        if recorder.create_episode():
            recording = True
            step_count = 0
            print("✅ episode recording started (record_on_start)")
        else:
            recording = False

    # Optional: local Wuji retargeter init (lazy)
    WujiHandRetargeter = None
    apply_mediapipe_transformations = None
    retargeter_left = None
    retargeter_right = None
    geort = None
    model_left = None
    model_right = None
    last_wuji_left = None
    last_wuji_right = None
    _warned_local_wuji = False

    try:
        while not stop_requested:
            t0 = time.time()
            rgb = vision.get_rgb()

            if not args.no_window:
                overlay = rgb.copy()
                status = "REC: ON" if recording else "REC: OFF"
                cv2.putText(
                    overlay,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if recording else (0, 0, 255),
                    2,
                )
                cv2.putText(overlay, "press r=start/stop, q=quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    recording = not recording
                    if recording:
                        if recorder.create_episode():
                            step_count = 0
                            print("✅ episode recording started")
                        else:
                            recording = False
                    else:
                        recorder.save_episode()
                        print("✅ episode saving triggered")

            if recording:
                data: Dict[str, Any] = {"idx": step_count}
                data["rgb"] = rgb
                data["t_img"] = now_ms()
                data["t_record_ms"] = now_ms()
                if bool(args.rs_depth):
                    depth = vision.get_depth()
                    data["depth"] = depth.tolist() if depth is not None else None

                try:
                    for k in redis_keys:
                        pipe.get(k)
                    results = pipe.execute()
                    for raw, dk in zip(results, data_dict_keys):
                        data[dk] = safe_json_loads(raw)
                except Exception as e:
                    print(f"⚠️ Redis read error: {e}")
                    continue


                # Local retarget: hand_tracking_* -> action_wuji_qpos_target_* (even without wuji_server)
                if bool(int(args.local_wuji_retarget)):
                    if WujiHandRetargeter is None or apply_mediapipe_transformations is None:
                        WujiHandRetargeter, apply_mediapipe_transformations = _try_import_wuji_retargeting()
                        if (WujiHandRetargeter is None or apply_mediapipe_transformations is None) and (not _warned_local_wuji):
                            _warned_local_wuji = True
                            print("[WARN] 本地 wuji retarget 初始化失败（将跳过 action_wuji_qpos_target_* 计算）。")
                    if bool(int(args.local_wuji_use_model)) and geort is None:
                        geort = _try_import_geort()
                        if geort is None and (not _warned_local_wuji):
                            _warned_local_wuji = True
                            print("[WARN] 本地 wuji model 推理初始化失败（无法 import geort；将跳过 action_wuji_qpos_target_* 计算）。")

                    def _maybe_retarget_one(side: str) -> None:
                        nonlocal retargeter_left, retargeter_right, geort, model_left, model_right, last_wuji_left, last_wuji_right
                        if WujiHandRetargeter is None or apply_mediapipe_transformations is None:
                            return
                        s = str(side).lower()
                        assert s in ["left", "right"]

                        key_ht = f"hand_tracking_{s}"
                        key_q = f"action_wuji_qpos_target_{s}"
                        key_t = f"t_action_wuji_hand_{s}"


                        # If already present and not overwriting, keep Redis value.
                        if  (not bool(int(args.local_wuji_retarget_overwrite))):
                            return


                        ht = data.get(key_ht, None)
                        if not isinstance(ht, dict):
                            return
                        if not bool(ht.get("is_active", False)):
                            return

                        hand_dict = {k: v for k, v in ht.items() if k not in ["is_active", "timestamp"]}
                        mp21 = hand_26d_to_mediapipe_21d(hand_dict, hand_side=s)
                        mp_trans = apply_mediapipe_transformations(mp21, hand_type=s)

                        # Choose DexPilot retarget (default) vs GeoRT model inference
                        if bool(int(args.local_wuji_use_model)):
                            if geort is None:
                                return
                            # per-side tag/epoch override
                            tag = str(args.local_wuji_policy_tag_left if s == "left" and str(args.local_wuji_policy_tag_left) else
                                      args.local_wuji_policy_tag_right if s == "right" and str(args.local_wuji_policy_tag_right) else
                                      args.local_wuji_policy_tag)
                            epoch_default = int(args.local_wuji_policy_epoch)
                            epoch_override = int(args.local_wuji_policy_epoch_left) if s == "left" else int(args.local_wuji_policy_epoch_right)
                            epoch = epoch_default if epoch_override == -999999 else epoch_override

                            if s == "left":
                                if model_left is None:
                                    print(f"[local_wuji:model] loading left: tag={tag}, epoch={epoch}")
                                    model_left = geort.load_model(tag, epoch=epoch)
                                    try:
                                        model_left.eval()
                                    except Exception:
                                        pass
                                model = model_left
                            else:
                                if model_right is None:
                                    print(f"[local_wuji:model] loading right: tag={tag}, epoch={epoch}")
                                    model_right = geort.load_model(tag, epoch=epoch)
                                    try:
                                        model_right.eval()
                                    except Exception:
                                        pass
                                model = model_right

                            pts21 = np.asarray(mp_trans, dtype=np.float32).reshape(21, 3)
                            if bool(int(args.local_wuji_use_fingertips5)):
                                human_points = pts21[[4, 8, 12, 16, 20], :3]  # (5,3)
                            else:
                                human_points = pts21
                            q = model.forward(human_points)
                            q = np.asarray(q, dtype=np.float32).reshape(-1)
                            if q.shape[0] != 20:
                                return
                            wuji_20d = q.reshape(5, 4)

                            # safety: clamp + rate limit
                            wuji_20d = np.clip(wuji_20d, float(args.local_wuji_clamp_min), float(args.local_wuji_clamp_max))
                            max_delta = float(args.local_wuji_max_delta_per_step)
                            if s == "left" and last_wuji_left is not None:
                                delta = wuji_20d - last_wuji_left
                                delta = np.clip(delta, -max_delta, max_delta)
                                wuji_20d = last_wuji_left + delta
                            if s == "right" and last_wuji_right is not None:
                                delta = wuji_20d - last_wuji_right
                                delta = np.clip(delta, -max_delta, max_delta)
                                wuji_20d = last_wuji_right + delta
                        else:
                            # DexPilot retargeter
                            if s == "left":
                                if retargeter_left is None:
                                    retargeter_left = WujiHandRetargeter(hand_side="left")
                                rr = retargeter_left.retarget(mp_trans)
                            else:
                                if retargeter_right is None:
                                    retargeter_right = WujiHandRetargeter(hand_side="right")
                                rr = retargeter_right.retarget(mp_trans)
                            wuji_20d = np.asarray(rr.robot_qpos, dtype=np.float32).reshape(5, 4)

                        if s == "left":
                            last_wuji_left = wuji_20d.copy()
                        else:
                            last_wuji_right = wuji_20d.copy()
                        now2 = now_ms()
                        data[key_q] = wuji_20d.reshape(-1).tolist()
                        data[key_t] = int(now2)


                        if bool(int(args.local_wuji_write_redis)):
                            try:
                                client.set(f"action_wuji_qpos_target_{s}_{suffix}", json.dumps(data[key_q]))
                                client.set(f"t_action_wuji_hand_{s}_{suffix}", int(now2))
                            except Exception:
                                pass

                    try:
                        _maybe_retarget_one("left")
                        _maybe_retarget_one("right")
                    except Exception:
                        # local retarget failure should not break recording
                        pass

                recorder.add_item(data)
                step_count += 1

                elapsed = time.time() - t0
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
            else:
                time.sleep(0.01)

    finally:
        try:
            if recording:
                recorder.save_episode()
        except Exception:
            pass
        try:
            recorder.close()
        except Exception:
            pass
        try:
            vision.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if xdm_proc is not None:
            try:
                os.killpg(os.getpgid(xdm_proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                xdm_proc.wait(timeout=1.0)
            except Exception:
                pass

    print(f"\nDone! Episodes saved under: {task_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


