#!/usr/bin/env python3
"""
Wuji Dual-Hand Controller via Redis (single process)

目的：
- 用一个 Python 进程同时控制左右 Wuji 手，避免双进程并发带来的 USB/调度抖动。
- 从 Redis 读取 hand_tracking_left/right_*（26D dict），分别 retarget，并下发到两只手。
- 可选平滑：采用“左右手同一步一起下发”的方式，保证总时长不变。

Redis keys（与 server_wuji_hand_redis.py 保持一致）：
- 输入：hand_tracking_{left/right}_unitree_g1_with_hands
- 输出（可用于录制）：action_wuji_qpos_target_{left/right}_unitree_g1_with_hands
                 state_wuji_hand_{left/right}_unitree_g1_with_hands
                 t_action_wuji_hand_{left/right}_unitree_g1_with_hands
                 t_state_wuji_hand_{left/right}_unitree_g1_with_hands
"""

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import redis

try:
    import wujihandpy
except ImportError:
    print("❌ 错误: 未安装 wujihandpy，请先安装:")
    print("   pip install wujihandpy")
    sys.exit(1)

# 复用单手脚本中的 retarget / mediapipe 转换逻辑，避免重复实现
try:
    from server_wuji_hand_redis import (
        apply_mediapipe_transformations,
        hand_26d_to_mediapipe_21d,
        now_ms,
        WujiHandRetargeter,
    )
except Exception as e:
    print(f"❌ 错误: 无法导入 server_wuji_hand_redis 中的依赖函数/类: {e}")
    print("   请确保 deploy_real/server_wuji_hand_redis.py 存在且 wuji_retargeting 已可导入")
    sys.exit(1)


def _safe_json_loads(data: Any) -> Optional[dict]:
    if data is None:
        return None
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)
    except Exception:
        return None


def _parse_tracking(hand_data: Optional[dict], stale_ms: int = 500) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Returns: (is_active_and_fresh, payload_dict_without_meta)
    """
    if not isinstance(hand_data, dict):
        return False, None
    ts = int(hand_data.get("timestamp", 0) or 0)
    is_active = bool(hand_data.get("is_active", False))
    if not is_active:
        return False, None
    diff = int(time.time() * 1000) - ts
    if diff > stale_ms:
        return False, None
    payload = {k: v for k, v in hand_data.items() if k not in ["timestamp", "is_active"]}
    return True, payload


@dataclass
class _HandCtx:
    side: str  # left|right
    serial_number: str
    robot_key: str = "unitree_g1_with_hands"
    hand: Optional["wujihandpy.Hand"] = None
    controller: Any = None
    retargeter: Any = None
    zero_pose: Optional[np.ndarray] = None
    last_qpos: Optional[np.ndarray] = None

    @property
    def redis_key_tracking(self) -> str:
        return f"hand_tracking_{self.side}_{self.robot_key}"

    @property
    def redis_key_action_qpos_target(self) -> str:
        return f"action_wuji_qpos_target_{self.side}_{self.robot_key}"

    @property
    def redis_key_state_qpos(self) -> str:
        return f"state_wuji_hand_{self.side}_{self.robot_key}"

    @property
    def redis_key_t_action(self) -> str:
        return f"t_action_wuji_hand_{self.side}_{self.robot_key}"

    @property
    def redis_key_t_state(self) -> str:
        return f"t_state_wuji_hand_{self.side}_{self.robot_key}"

    @property
    def redis_key_mode(self) -> str:
        # teleop 写入的模式开关：follow / hold / default
        return f"wuji_hand_mode_{self.side}_{self.robot_key}"


class WujiHandsRedisControllerDual:
    def __init__(
        self,
        redis_ip: str,
        target_fps: int,
        smooth_enabled: bool,
        smooth_steps: int,
        left_serial: str,
        right_serial: str,
    ):
        self.redis_ip = redis_ip
        self.target_fps = int(target_fps)
        self.control_dt = 1.0 / max(1, self.target_fps)
        self.smooth_enabled = bool(smooth_enabled)
        self.smooth_steps = int(max(1, smooth_steps))

        self.redis_client = redis.Redis(host=self.redis_ip, port=6379, decode_responses=False)
        self.redis_client.ping()

        self.left = _HandCtx(side="left", serial_number=(left_serial or "").strip())
        self.right = _HandCtx(side="right", serial_number=(right_serial or "").strip())

        self.running = True
        self._stop_requested_by_signal: Optional[int] = None
        self._cleaned_up = False

        self._init_hand(self.left)
        self._init_hand(self.right)

    def _init_hand(self, ctx: _HandCtx) -> None:
        print(f"🤖 初始化 Wuji {ctx.side} 手...")
        if not ctx.serial_number:
            raise ValueError(f"{ctx.side} 手未指定 serial_number（双手模式建议显式指定左右序列号）")

        print(f"🔌 {ctx.side} serial_number: {ctx.serial_number}")
        ctx.hand = wujihandpy.Hand(serial_number=ctx.serial_number)
        ctx.hand.write_joint_enabled(True)
        ctx.controller = ctx.hand.realtime_controller(
            enable_upstream=True,
            filter=wujihandpy.filter.LowPass(cutoff_freq=10.0),
        )
        time.sleep(0.4)

        # zero_pose 强制为全 0（与用户要求一致），但先读一次实际值以获得正确 shape
        actual_pose = ctx.hand.get_joint_actual_position()
        ctx.zero_pose = np.zeros_like(actual_pose)
        ctx.last_qpos = ctx.zero_pose.copy()

        print(f"🔄 初始化 WujiHandRetargeter ({ctx.side})...")
        ctx.retargeter = WujiHandRetargeter(hand_side=ctx.side)
        print(f"✅ Wuji {ctx.side} 手初始化完成")

    def _compute_target(self, ctx: _HandCtx, hand_payload: Dict[str, Any]) -> np.ndarray:
        mediapipe_21d = hand_26d_to_mediapipe_21d(hand_payload, ctx.side, print_distances=False)
        mediapipe_transformed = apply_mediapipe_transformations(mediapipe_21d, hand_type=ctx.side)
        retarget_result = ctx.retargeter.retarget(mediapipe_transformed)
        return retarget_result.robot_qpos.reshape(5, 4)

    def _set_target_no_smooth(self, left_q: Optional[np.ndarray], right_q: Optional[np.ndarray]) -> None:
        if left_q is not None and self.left.controller is not None:
            self.left.controller.set_joint_target_position(left_q)
        if right_q is not None and self.right.controller is not None:
            self.right.controller.set_joint_target_position(right_q)

    def _set_target_smooth(self, left_q: Optional[np.ndarray], right_q: Optional[np.ndarray]) -> None:
        # 左右手同一步一起下发，保证总时长 ~ control_dt，不会因为“顺序 smooth”变成 2x
        steps = int(max(1, self.smooth_steps))
        dt_per = self.control_dt / steps

        def _get_cur(ctrl: Any) -> np.ndarray:
            try:
                return ctrl.get_joint_actual_position()
            except Exception:
                return np.zeros((5, 4), dtype=np.float32)

        left_cur = _get_cur(self.left.controller) if (left_q is not None and self.left.controller is not None) else None
        right_cur = _get_cur(self.right.controller) if (right_q is not None and self.right.controller is not None) else None

        for t in np.linspace(0.0, 1.0, steps):
            if left_q is not None and left_cur is not None:
                ql = left_cur * (1 - t) + left_q * t
                self.left.controller.set_joint_target_position(ql)
            if right_q is not None and right_cur is not None:
                qr = right_cur * (1 - t) + right_q * t
                self.right.controller.set_joint_target_position(qr)
            time.sleep(dt_per)

    def run(self) -> None:
        def _handle_signal(signum, _frame):
            self._stop_requested_by_signal = signum
            self.running = False

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        print("=" * 70)
        print("Wuji Dual-Hand Controller via Redis (single process)")
        print("=" * 70)
        print(f"Redis IP: {self.redis_ip}")
        print(f"目标频率: {self.target_fps} Hz")
        print(f"平滑移动: {'启用' if self.smooth_enabled else '禁用'}")
        if self.smooth_enabled:
            print(f"平滑步数: {self.smooth_steps}")
        print(f"left_serial : {self.left.serial_number}")
        print(f"right_serial: {self.right.serial_number}")
        print("按 Ctrl+C 退出\n")

        # stats
        has_any_data = False
        debug_printed = {"stale": False, "no_key": False}

        try:
            while self.running:
                loop_start = time.time()

                pipe = self.redis_client.pipeline()
                pipe.get(self.left.redis_key_mode)
                pipe.get(self.right.redis_key_mode)
                pipe.get(self.left.redis_key_tracking)
                pipe.get(self.right.redis_key_tracking)
                left_mode_raw, right_mode_raw, left_raw, right_raw = pipe.execute()

                def _mode(x: Any) -> str:
                    if x is None:
                        return "follow"
                    if isinstance(x, bytes):
                        try:
                            x = x.decode("utf-8")
                        except Exception:
                            return "follow"
                    m = str(x).strip().lower()
                    return m if m in ["follow", "hold", "default"] else "follow"

                left_mode = _mode(left_mode_raw)
                right_mode = _mode(right_mode_raw)

                left_obj = _safe_json_loads(left_raw)
                right_obj = _safe_json_loads(right_raw)

                left_active, left_payload = _parse_tracking(left_obj, stale_ms=500)
                right_active, right_payload = _parse_tracking(right_obj, stale_ms=500)

                # 仅在 follow 模式下检查 tracking key 缺失（hold/default 不依赖 tracking）
                if (left_mode == "follow" or right_mode == "follow") and (left_raw is None or right_raw is None) and not debug_printed["no_key"]:
                    print(f"⚠️  Redis key 不存在或为空：{self.left.redis_key_tracking} / {self.right.redis_key_tracking}")
                    debug_printed["no_key"] = True

                if (left_mode == "follow" or right_mode == "follow") and (left_obj or right_obj) and (not left_active and not right_active) and not debug_printed["stale"]:
                    print("⚠️  双手追踪数据无效/过期（>500ms）或 is_active=False")
                    debug_printed["stale"] = True

                left_q = None
                right_q = None

                # 1) 模式优先
                if left_mode == "default":
                    left_q = self.left.zero_pose
                elif left_mode == "hold":
                    left_q = self.left.last_qpos
                elif left_mode == "follow":
                    if left_active and left_payload is not None:
                        left_q = self._compute_target(self.left, left_payload)

                if right_mode == "default":
                    right_q = self.right.zero_pose
                elif right_mode == "hold":
                    right_q = self.right.last_qpos
                elif right_mode == "follow":
                    if right_active and right_payload is not None:
                        right_q = self._compute_target(self.right, right_payload)

                if left_q is not None or right_q is not None:
                    has_any_data = True
                    # 写 action target（retarget 输出）
                    pipe = self.redis_client.pipeline()
                    if left_q is not None:
                        pipe.set(self.left.redis_key_action_qpos_target, json.dumps(left_q.reshape(-1).tolist()))
                        pipe.set(self.left.redis_key_t_action, now_ms())
                    if right_q is not None:
                        pipe.set(self.right.redis_key_action_qpos_target, json.dumps(right_q.reshape(-1).tolist()))
                        pipe.set(self.right.redis_key_t_action, now_ms())
                    pipe.execute()

                    # 下发控制
                    if self.smooth_enabled:
                        self._set_target_smooth(left_q, right_q)
                    else:
                        self._set_target_no_smooth(left_q, right_q)

                    # 写 state（硬件实际位置）
                    pipe = self.redis_client.pipeline()
                    if left_q is not None and self.left.hand is not None:
                        try:
                            actual = self.left.hand.get_joint_actual_position()
                            pipe.set(self.left.redis_key_state_qpos, json.dumps(actual.reshape(-1).tolist()))
                            pipe.set(self.left.redis_key_t_state, now_ms())
                        except Exception:
                            pass
                    if right_q is not None and self.right.hand is not None:
                        try:
                            actual = self.right.hand.get_joint_actual_position()
                            pipe.set(self.right.redis_key_state_qpos, json.dumps(actual.reshape(-1).tolist()))
                            pipe.set(self.right.redis_key_t_state, now_ms())
                        except Exception:
                            pass
                    pipe.execute()

                    if left_q is not None:
                        self.left.last_qpos = left_q.copy()
                    if right_q is not None:
                        self.right.last_qpos = right_q.copy()

                else:
                    if not has_any_data:
                        # 仅在起始阶段提示一次
                        pass

                # 控制频率
                elapsed = time.time() - loop_start
                if not self.smooth_enabled:
                    sleep_time = max(0.0, self.control_dt - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True

        print("\n🛑 正在关闭双手控制器并失能电机...")
        # SIGTERM 场景尽量快速回零，避免拖太久
        fast = self._stop_requested_by_signal == signal.SIGTERM
        duration = 0.2 if fast else 1.0
        steps = 10 if fast else 50

        try:
            # 同步回零（同一步一起下发）
            self._set_target_smooth(self.left.zero_pose, self.right.zero_pose) if self.smooth_enabled else self._set_target_no_smooth(self.left.zero_pose, self.right.zero_pose)
        except Exception:
            pass

        # 额外用一个“固定回零”保险（不依赖 smooth_enabled），减少残留状态
        try:
            dt_per = duration / steps
            left_cur = self.left.controller.get_joint_actual_position() if self.left.controller is not None else None
            right_cur = self.right.controller.get_joint_actual_position() if self.right.controller is not None else None
            for t in np.linspace(0.0, 1.0, steps):
                if left_cur is not None and self.left.zero_pose is not None and self.left.controller is not None:
                    ql = left_cur * (1 - t) + self.left.zero_pose * t
                    self.left.controller.set_joint_target_position(ql)
                if right_cur is not None and self.right.zero_pose is not None and self.right.controller is not None:
                    qr = right_cur * (1 - t) + self.right.zero_pose * t
                    self.right.controller.set_joint_target_position(qr)
                time.sleep(dt_per)
        except Exception:
            pass

        for ctx in [self.left, self.right]:
            try:
                if ctx.controller is not None:
                    ctx.controller.close()
                if ctx.hand is not None:
                    ctx.hand.write_joint_enabled(False)
                print(f"✅ {ctx.side} 已关闭")
            except Exception:
                pass

        print("✅ 退出完成")


def parse_args():
    p = argparse.ArgumentParser(description="单进程双手 Wuji Redis 控制器")
    p.add_argument("--redis_ip", type=str, default="localhost", help="Redis IP (默认: localhost)")
    p.add_argument("--target_fps", type=int, default=50, help="目标频率 Hz (默认: 50)")
    p.add_argument("--no_smooth", action="store_true", help="禁用平滑移动")
    p.add_argument("--smooth_steps", type=int, default=5, help="平滑步数 (默认: 5)")
    p.add_argument("--left_serial", type=str, default="", help="左手 serial_number（必填）")
    p.add_argument("--right_serial", type=str, default="", help="右手 serial_number（必填）")
    p.add_argument("--serial_number", type=str, default="", help="便捷：同时给左右手指定同一个 serial（通常不用）")
    return p.parse_args()


def main():
    args = parse_args()
    left_serial = args.left_serial.strip() or args.serial_number.strip()
    right_serial = args.right_serial.strip() or args.serial_number.strip()
    if not left_serial or not right_serial:
        print("❌ 错误: 双手模式必须提供 --left_serial 和 --right_serial（或用 --serial_number 同时指定）")
        sys.exit(2)

    ctrl = WujiHandsRedisControllerDual(
        redis_ip=args.redis_ip,
        target_fps=args.target_fps,
        smooth_enabled=not args.no_smooth,
        smooth_steps=args.smooth_steps,
        left_serial=left_serial,
        right_serial=right_serial,
    )
    ctrl.run()


if __name__ == "__main__":
    main()


