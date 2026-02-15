#!/usr/bin/env python3
"""
Wuji Hand SIM Visualization via Redis

目标：
- 从 Redis 读取 xdmocap_teleop_body_to_redis.py 写入的 hand_tracking_*（26D dict）
- 按 deploy_real/server_wuji_hand_redis.py 的后处理方式：
  26D -> 21D MediaPipe -> apply_mediapipe_transformations -> WujiHandRetargeter
- 将 retarget 输出映射到 MuJoCo 手模型（wuji_retargeting/example/utils/mujoco-sim/model/{left,right}.xml）
  并在 viewer 中实时可视化

说明：
- 本脚本只做 sim 可视化，不依赖/不控制真实 wujihandpy 硬件
- 同时支持 follow/hold/default 三种模式（读取 wuji_hand_mode_*）
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np  # type: ignore

try:
    import redis  # type: ignore
except Exception as e:
    raise SystemExit(
        "❌ 缺少依赖 `redis`（python-redis）。\n"
        "   解决：在你的运行环境里安装，例如：\n"
        "     - pip install redis\n"
        "     - 或 conda install -c conda-forge redis-py\n"
        f"   原始错误：{e}"
    )


def now_ms() -> int:
    return int(time.time() * 1000)


# 26维手部关节名称（与 deploy_real/server_wuji_hand_redis.py 一致）
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


# 26D -> 21D MediaPipe mapping（与 deploy_real/server_wuji_hand_redis.py 一致）
MEDIAPIPE_MAPPING_26_TO_21 = [
    1,  # 0: Wrist -> Wrist
    2,  # 1: ThumbMetacarpal -> Thumb CMC
    3,  # 2: ThumbProximal -> Thumb MCP
    4,  # 3: ThumbDistal -> Thumb IP
    5,  # 4: ThumbTip -> Thumb Tip
    6,  # 5: IndexMetacarpal -> Index MCP
    7,  # 6: IndexProximal -> Index PIP
    8,  # 7: IndexIntermediate -> Index DIP
    10,  # 8: IndexTip -> Index Tip (skip IndexDistal)
    11,  # 9: MiddleMetacarpal -> Middle MCP
    12,  # 10: MiddleProximal -> Middle PIP
    13,  # 11: MiddleIntermediate -> Middle DIP
    15,  # 12: MiddleTip -> Middle Tip (skip MiddleDistal)
    16,  # 13: RingMetacarpal -> Ring MCP
    17,  # 14: RingProximal -> Ring PIP
    18,  # 15: RingIntermediate -> Ring DIP
    20,  # 16: RingTip -> Ring Tip (skip RingDistal)
    21,  # 17: LittleMetacarpal -> Pinky MCP
    22,  # 18: LittleProximal -> Pinky PIP
    23,  # 19: LittleIntermediate -> Pinky DIP
    25,  # 20: LittleTip -> Pinky Tip (skip LittleDistal)
]


def hand_26d_to_mediapipe_21d(hand_data_dict: Dict[str, Any], hand_side: str) -> np.ndarray:
    """
    将 26D dict（key: LeftHandWrist/RightHandWrist...）转换为 (21,3) MediaPipe 关键点。
    与 deploy_real/server_wuji_hand_redis.py 的逻辑保持一致：
    - 取 position
    - wrist 作为原点（整体减 wrist）
    """
    side = hand_side.strip().lower()
    assert side in ["left", "right"]
    hand_side_prefix = "LeftHand" if side == "left" else "RightHand"

    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_data_dict:
            pos = hand_data_dict[key][0]
            joint_positions_26[i] = np.asarray(pos, dtype=np.float32).reshape(3)
        else:
            joint_positions_26[i] = np.zeros(3, dtype=np.float32)

    mp21 = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21].astype(np.float32)
    wrist_pos = mp21[0].copy()
    mp21 = mp21 - wrist_pos
    return mp21


def _build_wuji_reorder_idx(retargeter: Any) -> Optional[np.ndarray]:
    """
    retarget_result.robot_qpos 的顺序不一定是 finger{i}_joint{j} 的自然顺序；
    这里参考 deploy_real/server_wuji_hand_redis.py：按 joint name 重排到：
      finger1_joint1..4, finger2_joint1..4, ... finger5_joint1..4
    """
    try:
        opt = getattr(retargeter, "optimizer", None)
        names = list(getattr(opt, "target_joint_names", [])) if opt is not None else []
    except Exception:
        names = []

    if not names:
        return None

    desired_joint_names = [f"finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]
    name2idx = {n: i for i, n in enumerate(names)}
    if not all(n in name2idx for n in desired_joint_names):
        return None
    return np.array([name2idx[n] for n in desired_joint_names], dtype=np.int32)


class WujiHandSimRedisViz:
    def __init__(
        self,
        *,
        redis_ip: str,
        hand_side: str,
        target_fps: float,
        freshness_ms: int,
        mjcf_path: Optional[str],
        write_action_target: bool,
        verbose: bool,
        # mode switch: DexPilot retarget (default) vs GeoRT model inference
        use_model: bool,
        policy_tag: str,
        policy_epoch: int,
        use_fingertips5: bool,
        clamp_min: float,
        clamp_max: float,
        max_delta_per_step: float,
    ) -> None:
        self.hand_side = hand_side.strip().lower()
        assert self.hand_side in ["left", "right"]
        self.target_fps = float(target_fps)
        self.freshness_ms = int(freshness_ms)
        self.write_action_target = bool(write_action_target)
        self.verbose = bool(verbose)
        self.use_model = bool(use_model)
        self.policy_tag = str(policy_tag)
        self.policy_epoch = int(policy_epoch)
        self.use_fingertips5 = bool(use_fingertips5)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.max_delta_per_step = float(max_delta_per_step)

        self.running = True
        self._stop_requested_by_signal: Optional[int] = None

        # Redis keys (keep consistent with deploy_real/server_wuji_hand_redis.py)
        self.robot_key = "unitree_g1_with_hands"
        self.redis_key_hand_tracking = f"hand_tracking_{self.hand_side}_{self.robot_key}"
        self.redis_key_action_wuji_qpos_target = f"action_wuji_qpos_target_{self.hand_side}_{self.robot_key}"
        self.redis_key_t_action_wuji_hand = f"t_action_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_wuji_mode = f"wuji_hand_mode_{self.hand_side}_{self.robot_key}"

        print(f"[WujiHandSim] connecting redis {redis_ip}:6379 ...")
        self.redis_client = redis.Redis(host=str(redis_ip), port=6379, decode_responses=False)
        self.redis_client.ping()
        print("[WujiHandSim] redis ok")

        # Import wuji_retargeting (no wujihandpy dependency)
        project_root = Path(__file__).resolve().parents[1]
        wuji_retargeting_path = project_root / "wuji_retargeting"
        if str(wuji_retargeting_path) not in sys.path:
            sys.path.insert(0, str(wuji_retargeting_path))
        # NOTE: `wuji_retargeting/__init__.py` will set WujiHandRetargeter=None if Pinocchio is missing.
        # For SIM we need the real class, so import from the implementation module directly.
        try:
            from wuji_retargeting.retarget import WujiHandRetargeter  # type: ignore
            from wuji_retargeting.mediapipe import apply_mediapipe_transformations  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "无法导入 wuji_retargeting 的 retarget 模块（通常是缺少 pinocchio）。\n"
                "解决：在运行环境中安装 pinocchio，例如：\n"
                "  conda install -c conda-forge pinocchio\n"
                f"原始错误：{repr(e)}"
            ) from e

        self._apply_mediapipe_transformations = apply_mediapipe_transformations
        self.retargeter = None
        self._wuji_reorder_idx = None
        self.model_infer = None

        if self.use_model:
            # GeoRT model inference (same API as deploy2.py)
            try:
                wuji_retarget_path = project_root / "wuji_retarget"
                if str(wuji_retarget_path) not in sys.path:
                    sys.path.insert(0, str(wuji_retarget_path))
                import geort  # type: ignore
            except Exception as e:
                raise RuntimeError(f"无法导入 geort（需要 wuji_retarget/ 在 PYTHONPATH）。错误：{e}") from e

            print(f"[WujiHandSim] 🔄 load GeoRT model: tag={self.policy_tag}, epoch={self.policy_epoch}")
            self.model_infer = geort.load_model(self.policy_tag, epoch=self.policy_epoch)
            try:
                self.model_infer.eval()
            except Exception:
                pass
            print("[WujiHandSim] ✅ GeoRT model loaded")
        else:
            self.retargeter = WujiHandRetargeter(hand_side=self.hand_side)
            self._wuji_reorder_idx = _build_wuji_reorder_idx(self.retargeter)
            if self._wuji_reorder_idx is None:
                print("[WujiHandSim] ⚠️ 未能通过 joint name 推断重排索引，将回退为 reshape(5,4)")

        # Load MuJoCo model
        try:
            import mujoco  # type: ignore
            import mujoco.viewer  # type: ignore
        except Exception as e:
            raise RuntimeError(f"未安装 mujoco 或 GUI 依赖异常：{e}") from e

        self._mujoco = mujoco
        self._mjviewer = mujoco.viewer

        if mjcf_path is None or str(mjcf_path).strip() == "":
            mjcf_path = str(
                (project_root / "wuji_retargeting" / "example" / "utils" / "mujoco-sim" / "model" / f"{self.hand_side}.xml").resolve()
            )
        self.mjcf_path = str(mjcf_path)
        if not Path(self.mjcf_path).exists():
            raise FileNotFoundError(f"MuJoCo model file not found: {self.mjcf_path}")

        self.model = self._mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = self._mujoco.MjData(self.model)

        # init ctrl to mid-range for stability (same as wuji_retargeting sim)
        for i in range(int(self.model.nu)):
            if bool(self.model.actuator_ctrllimited[i]):
                lo, hi = self.model.actuator_ctrlrange[i]
                self.data.ctrl[i] = 0.5 * (float(lo) + float(hi))
            else:
                self.data.ctrl[i] = 0.0

        for _ in range(100):
            self._mujoco.mj_step(self.model, self.data)

        self.zero_pose = np.zeros((5, 4), dtype=np.float32)
        self.last_qpos = self.zero_pose.copy()

    def _get_mode(self) -> str:
        try:
            mode_raw = self.redis_client.get(self.redis_key_wuji_mode)
            if isinstance(mode_raw, bytes):
                mode_raw = mode_raw.decode("utf-8", errors="ignore")
            mode = str(mode_raw) if mode_raw is not None else "follow"
        except Exception:
            mode = "follow"
        return mode.strip().lower()

    def _read_tracking26(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        返回 (ok, dict)：
        - ok=True 且 dict != None: 数据新鲜且 is_active=True
        - ok=False: 无数据/过期/不 active
        """
        try:
            raw = self.redis_client.get(self.redis_key_hand_tracking)
            if raw is None:
                return False, None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return False, None
            ts = int(payload.get("timestamp", 0))
            if (now_ms() - ts) > int(self.freshness_ms):
                return False, None
            if not bool(payload.get("is_active", False)):
                return False, None
            hand_dict = {k: v for k, v in payload.items() if k not in ["is_active", "timestamp"]}
            return True, hand_dict
        except Exception:
            return False, None

    def _retarget_to_wuji20(self, hand_dict: Dict[str, Any]) -> np.ndarray:
        mp21 = hand_26d_to_mediapipe_21d(hand_dict, self.hand_side)
        mp21_t = self._apply_mediapipe_transformations(mp21, hand_type=self.hand_side)
        if self.use_model:
            if self.model_infer is None:
                raise RuntimeError("model_infer is None but use_model=True")
            pts21 = np.asarray(mp21_t, dtype=np.float32).reshape(21, 3)
            if self.use_fingertips5:
                human_points = pts21[[4, 8, 12, 16, 20], :3]
            else:
                human_points = pts21
            q = self.model_infer.forward(human_points)
            q = np.asarray(q, dtype=np.float32).reshape(-1)
            if q.shape[0] != 20:
                raise ValueError(f"Model output dim mismatch: expect 20, got {q.shape[0]}")
            return q.reshape(5, 4)
        else:
            if self.retargeter is None:
                raise RuntimeError("retargeter is None but use_model=False")
            rr = self.retargeter.retarget(mp21_t)
            qpos = np.asarray(rr.robot_qpos, dtype=np.float32).reshape(-1)
            if self._wuji_reorder_idx is not None and qpos.shape[0] >= int(self._wuji_reorder_idx.max() + 1):
                qpos = qpos[self._wuji_reorder_idx]
            return qpos.reshape(5, 4)

    def _apply_safety(self, qpos_5x4: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos_5x4, dtype=np.float32).reshape(5, 4)
        q = np.clip(q, self.clamp_min, self.clamp_max)
        if self.last_qpos is not None and np.asarray(self.last_qpos).shape == q.shape:
            d = q - self.last_qpos
            d = np.clip(d, -self.max_delta_per_step, self.max_delta_per_step)
            q = self.last_qpos + d
        return q

    def _publish_action_target(self, qpos_5x4: np.ndarray) -> None:
        if not self.write_action_target:
            return
        try:
            self.redis_client.set(self.redis_key_action_wuji_qpos_target, json.dumps(np.asarray(qpos_5x4, dtype=float).reshape(-1).tolist()))
            self.redis_client.set(self.redis_key_t_action_wuji_hand, now_ms())
        except Exception:
            pass

    def run(self) -> int:
        def _handle_signal(signum: int, _frame: Any) -> None:
            self._stop_requested_by_signal = int(signum)
            self.running = False

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        print("=" * 70)
        print(f"[WujiHandSim] MuJoCo viewer for {self.hand_side} hand")
        print(f"[WujiHandSim] mjcf: {self.mjcf_path}")
        print(f"[WujiHandSim] redis: {self.redis_client.connection_pool.connection_kwargs.get('host', '')}:6379")
        print(f"[WujiHandSim] keys : {self.redis_key_hand_tracking} / {self.redis_key_wuji_mode}")
        print("=" * 70)

        ctrl_dt = 1.0 / max(1.0, float(self.target_fps))
        sim_dt = float(self.model.opt.timestep)
        steps_per_ctrl = max(1, int(round(ctrl_dt / max(1e-6, sim_dt))))

        # launch viewer (passive)
        try:
            with self._mjviewer.launch_passive(self.model, self.data) as viewer:
                # camera pose like wuji_retargeting sim
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -20
                viewer.cam.distance = 0.5
                viewer.cam.lookat[:] = [0, 0, 0.05]

                last_print = 0.0
                while self.running and viewer.is_running():
                    t0 = time.time()

                    mode = self._get_mode()
                    if mode in ["default", "hold"]:
                        target = self.zero_pose if mode == "default" else self.last_qpos
                        self.last_qpos = np.asarray(target, dtype=np.float32).reshape(5, 4).copy()
                        self._publish_action_target(self.last_qpos)
                    else:
                        # 1) Preferred: retarget from hand_tracking_* (requires pinocchio)
                        ok, hand_dict = self._read_tracking26()
                        if ok and hand_dict is not None:
                            try:
                                q = self._retarget_to_wuji20(hand_dict).astype(np.float32)
                                if self.use_model:
                                    q = self._apply_safety(q)
                                self.last_qpos = q
                                self._publish_action_target(self.last_qpos)
                            except Exception as e:
                                if self.verbose:
                                    print(f"[WujiHandSim] ⚠️ retarget failed: {e}")

                    flat = self.last_qpos.reshape(-1)
                    n = min(int(self.model.nu), int(flat.shape[0]))
                    self.data.ctrl[:n] = flat[:n]

                    for _ in range(steps_per_ctrl):
                        self._mujoco.mj_step(self.model, self.data)

                    viewer.sync()

                    # light status print
                    if self.verbose and (time.time() - last_print) > 2.0:
                        last_print = time.time()
                        print(f"[WujiHandSim] mode={mode} fps={1.0/max(1e-6, (time.time()-t0)):.1f} steps={steps_per_ctrl}")

                    # rate limit
                    elapsed = time.time() - t0
                    sleep_s = max(0.0, ctrl_dt - elapsed)
                    if sleep_s > 0:
                        time.sleep(sleep_s)

        except Exception as e:
            print(f"[WujiHandSim] ❌ viewer 启动失败（可能无 GUI / DISPLAY 未设置）：{e}")
            return 2

        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wuji hand MuJoCo sim visualization (Redis -> Retarget -> MuJoCo)")
    p.add_argument("--hand_side", type=str, default="left", choices=["left", "right"], help="left/right")
    p.add_argument("--redis_ip", type=str, default="localhost", help="Redis host")
    p.add_argument("--target_fps", type=float, default=60.0, help="控制/渲染更新频率（Hz）")
    p.add_argument("--freshness_ms", type=int, default=500, help="hand_tracking 数据新鲜度阈值（毫秒）")
    p.add_argument("--mjcf_path", type=str, default="", help="可选：手模型 xml 路径（默认使用 wuji_retargeting/example/utils/mujoco-sim/model/{side}.xml）")
    p.add_argument("--no_write_action_target", action="store_true", help="不写 action_wuji_qpos_target_* 回 Redis（默认会写，便于排障/录制）")
    p.add_argument("--verbose", action="store_true", help="打印更多日志")
    # mode switch (align with deploy2.py / wuji_hand_model_deploy.sh)
    p.add_argument("--use_model", action="store_true", help="使用 GeoRT 模型推理（默认不启用，使用 DexPilot retarget）")
    p.add_argument("--policy_tag", type=str, default="geort_filter_wuji", help="GeoRT 模型 tag（--use_model）")
    p.add_argument("--policy_epoch", type=int, default=-1, help="GeoRT 模型 epoch（--use_model）")
    p.add_argument("--use_fingertips5", action="store_true", help="model 输入用 5 指尖（默认启用）")
    p.set_defaults(use_fingertips5=True)
    p.add_argument("--clamp_min", type=float, default=-1.5, help="model 输出限幅最小值")
    p.add_argument("--clamp_max", type=float, default=1.5, help="model 输出限幅最大值")
    p.add_argument("--max_delta_per_step", type=float, default=0.08, help="model 输出每步最大变化")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    viz = WujiHandSimRedisViz(
        redis_ip=str(args.redis_ip),
        hand_side=str(args.hand_side),
        target_fps=float(args.target_fps),
        freshness_ms=int(args.freshness_ms),
        mjcf_path=(str(args.mjcf_path).strip() or None),
        write_action_target=not bool(args.no_write_action_target),
        verbose=bool(args.verbose),
        use_model=bool(args.use_model),
        policy_tag=str(args.policy_tag),
        policy_epoch=int(args.policy_epoch),
        use_fingertips5=bool(args.use_fingertips5),
        clamp_min=float(args.clamp_min),
        clamp_max=float(args.clamp_max),
        max_delta_per_step=float(args.max_delta_per_step),
    )
    return int(viz.run())


if __name__ == "__main__":
    raise SystemExit(main())


