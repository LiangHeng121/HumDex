#!/usr/bin/env python3
"""
从 Redis 抓取“当前姿态”（全身 + Unitree 夹爪手 + 脖子 + Wuji 灵巧手）并打印出来，
便于你修改 teleop 按键 k/p 时的默认/保持逻辑。

默认读取 teleop 正在发布的 action_* keys（推荐）：
  - action_body_unitree_g1_with_hands (35D)
  - action_hand_left_unitree_g1_with_hands (7D)
  - action_hand_right_unitree_g1_with_hands (7D)
  - action_neck_unitree_g1_with_hands (2D)

额外会读取（与 source 无关）：
  - hand_tracking_left/right_unitree_g1_with_hands（teleop -> Wuji 的输入）
  - wuji_hand_mode_left/right_unitree_g1_with_hands（teleop -> Wuji 的模式：follow/hold/default）
  - action_wuji_qpos_target_left/right_unitree_g1_with_hands（Wuji retarget 输出的目标 20D）
  - state_wuji_hand_left/right_unitree_g1_with_hands（Wuji 硬件实际 20D）

也支持读取机器人反馈 state_*（如果你更想用“实际状态”而不是“指令目标”）。

另外支持 **不依赖 Redis**：直接从 G1 读取当前状态并拼成 35D mimic_obs（用于你想“按当前真机姿态当 default”的场景）。
注意：该模式需要在能直连 G1 的环境运行（通常是 g1 本机或同网段可访问机器），且依赖 `deploy_real/robot_control/*` 的 Unitree 接口。

用法示例：
  python deploy_real/capture_current_pose_from_redis.py --redis_ip localhost --source action
  python deploy_real/capture_current_pose_from_redis.py --redis_ip 192.168.123.222 --source state
  python deploy_real/capture_current_pose_from_redis.py --source g1 --net eno1
"""

import argparse
import json
import sys
from typing import Any, Optional, List

import redis


def _decode_if_bytes(x: Any) -> Any:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def _get_json_value(client: "redis.Redis", key: str) -> Optional[Any]:
    raw = client.get(key)
    if raw is None:
        return None
    raw = _decode_if_bytes(raw)
    try:
        return json.loads(raw)
    except Exception:
        return None


def _assert_list_len(name: str, v: Any, expected_len: int) -> list:
    if not isinstance(v, list):
        raise ValueError(f"{name} 不是 list，实际类型: {type(v)}")
    if len(v) != expected_len:
        raise ValueError(f"{name} 长度不对，期望 {expected_len}，实际 {len(v)}")
    return v


def _fmt_python_list(v: list) -> str:
    # 更适合复制粘贴：紧凑但可读
    return json.dumps(v, ensure_ascii=False)

def _fmt_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def _parse_neck_2(s: str) -> List[float]:
    # Accept "0,0" or "0 0"
    raw = str(s).strip().replace(",", " ")
    parts = [p for p in raw.split() if p]
    if len(parts) != 2:
        raise ValueError(f"--neck 期望 2 个数字，例如 '0,0'，实际: {s!r}")
    return [float(parts[0]), float(parts[1])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis IP/Host")
    parser.add_argument(
        "--source",
        type=str,
        default="action",
        choices=["action", "state", "g1"],
        help="action/state：从 Redis 读 action_*/state_*；g1：不依赖 Redis，直读 G1 状态并拼成 35D mimic_obs",
    )
    parser.add_argument("--net", type=str, default="eno1", help="G1 网络接口（source=g1 时使用）")
    parser.add_argument(
        "--g1_config",
        type=str,
        default="deploy_real/robot_control/configs/g1.yaml",
        help="G1 配置文件（source=g1 时使用）",
    )
    parser.add_argument(
        "--g1_height",
        type=float,
        default=0.79,
        help="拼 35D mimic_obs 时的 root_pos_z（source=g1 时使用；真实高度难以直接读，默认 0.79）",
    )
    parser.add_argument(
        "--use_hand",
        action="store_true",
        help="source=g1 时同时直读 Unitree 7D 夹爪手（Dex3_1），作为 HAND_LEFT/RIGHT_7 输出",
    )
    parser.add_argument(
        "--neck",
        type=str,
        default="0,0",
        help="source=g1 时输出的 NECK_2（默认 0,0；格式: 'yaw,pitch'）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格校验维度（35/7/7/2），否则只要能解析就打印。",
    )
    args = parser.parse_args()

    # ----------------------------
    # Mode: direct from G1 (no Redis required)
    # ----------------------------
    if args.source == "g1":
        try:
            # Lazy imports so non-G1 machines can still run Redis modes without Unitree deps.
            # When running `python deploy_real/xxx.py`, `deploy_real/` is on sys.path,
            # so we import like other deploy_real scripts do.
            from robot_control.g1_wrapper import G1RealWorldEnv
            from robot_control.config import Config as G1Config
            from data_utils.rot_utils import quatToEuler
        except Exception as e:
            print(
                "[ERROR] source=g1 需要 Unitree/G1 相关依赖，但导入失败。\n"
                "请确认你在 g1 本机环境运行，且依赖已安装；或改用 --source action/state 从 Redis 读取。\n"
                f"导入错误: {e}",
                file=sys.stderr,
            )
            return 2

        neck = _parse_neck_2(args.neck)
        cfg = G1Config(args.g1_config)
        env = None
        hand_ctrl = None
        try:
            env = G1RealWorldEnv(net=args.net, config=cfg)

            # Read twice to avoid occasional first-frame weirdness.
            _ = env.get_robot_state()
            dof_pos, _dof_vel, quat, ang_vel, *_rest = env.get_robot_state()

            rpy = quatToEuler(quat)  # [roll, pitch, yaw]
            roll = float(rpy[0])
            pitch = float(rpy[1])
            yaw_ang_vel = float(ang_vel[2]) if hasattr(ang_vel, "__len__") and len(ang_vel) >= 3 else 0.0

            body_35 = [0.0, 0.0, float(args.g1_height), roll, pitch, yaw_ang_vel] + list(
                map(float, dof_pos.tolist())
            )
            if args.strict:
                _assert_list_len("BODY_35(g1)", body_35, 35)

            hand_left = None
            hand_right = None
            if args.use_hand:
                try:
                    from robot_control.dex_hand_wrapper import Dex3_1_Controller

                    hand_ctrl = Dex3_1_Controller(args.net, re_init=False)
                    lh, rh = hand_ctrl.get_hand_state()
                    hand_left = list(map(float, lh.tolist()))
                    hand_right = list(map(float, rh.tolist()))
                    if args.strict:
                        _assert_list_len("HAND_LEFT_7(g1)", hand_left, 7)
                        _assert_list_len("HAND_RIGHT_7(g1)", hand_right, 7)
                except Exception as e:
                    print(f"[WARN] source=g1 --use_hand 读取 7D 夹爪手失败（将不输出 HAND_*_7）：{e}", file=sys.stderr)

            print("=== Capture Current Pose (Direct G1) ===")
            print(f"- net: {args.net}")
            print(f"- g1_config: {args.g1_config}")
            print(f"- g1_height: {args.g1_height}")
            print("")

            print("BODY_35 = " + _fmt_python_list(body_35))
            if hand_left is not None:
                print("HAND_LEFT_7 = " + _fmt_python_list(hand_left))
            if hand_right is not None:
                print("HAND_RIGHT_7 = " + _fmt_python_list(hand_right))
            print("NECK_2 = " + _fmt_python_list(neck))
            return 0
        finally:
            try:
                if hand_ctrl is not None:
                    hand_ctrl.close()
            except Exception:
                pass
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass

    # ----------------------------
    # Mode: from Redis (action/state)
    # ----------------------------
    # 使用 decode_responses=False 与 teleop / wuji hand 脚本保持一致
    client = redis.Redis(host=args.redis_ip, port=6379, db=0, decode_responses=False)

    prefix = "action" if args.source == "action" else "state"
    keys = {
        "body_35": f"{prefix}_body_unitree_g1_with_hands",
        "hand_left_7": f"{prefix}_hand_left_unitree_g1_with_hands",
        "hand_right_7": f"{prefix}_hand_right_unitree_g1_with_hands",
        "neck_2": f"{prefix}_neck_unitree_g1_with_hands",
        "t_ms": f"t_{prefix}",
    }

    try:
        client.ping()
    except Exception as e:
        print(f"[ERROR] Redis 连接失败: {e}", file=sys.stderr)
        return 2

    body = _get_json_value(client, keys["body_35"])
    hand_left = _get_json_value(client, keys["hand_left_7"])
    hand_right = _get_json_value(client, keys["hand_right_7"])
    neck = _get_json_value(client, keys["neck_2"])
    t_ms = _get_json_value(client, keys["t_ms"])

    # Teleop -> Wuji 的手部追踪输入（26D dict + is_active/timestamp）
    ht_left_key = "hand_tracking_left_unitree_g1_with_hands"
    ht_right_key = "hand_tracking_right_unitree_g1_with_hands"
    mode_left_key = "wuji_hand_mode_left_unitree_g1_with_hands"
    mode_right_key = "wuji_hand_mode_right_unitree_g1_with_hands"
    hand_tracking_left = _get_json_value(client, ht_left_key)
    hand_tracking_right = _get_json_value(client, ht_right_key)
    wuji_mode_left = _get_json_value(client, mode_left_key)
    wuji_mode_right = _get_json_value(client, mode_right_key)

    # Wuji 输出：目标与实际（20D）
    wuji_keys = {
        "action_wuji_left_20": "action_wuji_qpos_target_left_unitree_g1_with_hands",
        "action_wuji_right_20": "action_wuji_qpos_target_right_unitree_g1_with_hands",
        "state_wuji_left_20": "state_wuji_hand_left_unitree_g1_with_hands",
        "state_wuji_right_20": "state_wuji_hand_right_unitree_g1_with_hands",
        "t_action_wuji_left": "t_action_wuji_hand_left_unitree_g1_with_hands",
        "t_action_wuji_right": "t_action_wuji_hand_right_unitree_g1_with_hands",
        "t_state_wuji_left": "t_state_wuji_hand_left_unitree_g1_with_hands",
        "t_state_wuji_right": "t_state_wuji_hand_right_unitree_g1_with_hands",
    }
    wuji = {k: _get_json_value(client, redis_key) for k, redis_key in wuji_keys.items()}

    missing = [k for k, redis_key in keys.items() if k != "t_ms" and _get_json_value(client, redis_key) is None]
    if (
        body is None
        and hand_left is None
        and hand_right is None
        and neck is None
        and hand_tracking_left is None
        and hand_tracking_right is None
        and all(v is None for v in wuji.values())
    ):
        print("[ERROR] 没有读到任何姿态数据。你可能还没启动 teleop/sim2real，或 key 名不匹配。", file=sys.stderr)
        print(f"  - 建议先确认 Redis 里有这些 key（source={args.source}）：", file=sys.stderr)
        for _, redis_key in keys.items():
            print(f"    - {redis_key}", file=sys.stderr)
        print("  - 以及这些额外 key：", file=sys.stderr)
        print(f"    - {ht_left_key}", file=sys.stderr)
        print(f"    - {ht_right_key}", file=sys.stderr)
        for _, redis_key in wuji_keys.items():
            print(f"    - {redis_key}", file=sys.stderr)
        return 3

    # 维度校验（可选）
    if args.strict:
        if body is not None:
            body = _assert_list_len(keys["body_35"], body, 35)
        if hand_left is not None:
            hand_left = _assert_list_len(keys["hand_left_7"], hand_left, 7)
        if hand_right is not None:
            hand_right = _assert_list_len(keys["hand_right_7"], hand_right, 7)
        if neck is not None:
            neck = _assert_list_len(keys["neck_2"], neck, 2)
        # Wuji 20D 校验（如果存在）
        for k in ["action_wuji_left_20", "action_wuji_right_20", "state_wuji_left_20", "state_wuji_right_20"]:
            if wuji.get(k) is not None:
                _assert_list_len(wuji_keys[k], wuji[k], 20)

    print("=== Capture Current Pose From Redis ===")
    print(f"- redis_ip: {args.redis_ip}")
    print(f"- source: {args.source}")
    print(f"- t_{prefix}: {t_ms}")
    if missing:
        print(f"- missing_keys: {missing}")
    print("")

    if body is not None:
        print("BODY_35 = " + _fmt_python_list(body))
    if hand_left is not None:
        print("HAND_LEFT_7 = " + _fmt_python_list(hand_left))
    if hand_right is not None:
        print("HAND_RIGHT_7 = " + _fmt_python_list(hand_right))
    if neck is not None:
        print("NECK_2 = " + _fmt_python_list(neck))

    if hand_tracking_left is not None:
        print("HAND_TRACKING_LEFT = " + _fmt_json(hand_tracking_left))
    if hand_tracking_right is not None:
        print("HAND_TRACKING_RIGHT = " + _fmt_json(hand_tracking_right))
    if wuji_mode_left is not None or wuji_mode_right is not None:
        print(f"WUJI_MODE = left:{wuji_mode_left} right:{wuji_mode_right}")

    # Wuji
    if wuji.get("action_wuji_left_20") is not None:
        print("WUJI_ACTION_LEFT_20 = " + _fmt_python_list(wuji["action_wuji_left_20"]))
    if wuji.get("action_wuji_right_20") is not None:
        print("WUJI_ACTION_RIGHT_20 = " + _fmt_python_list(wuji["action_wuji_right_20"]))
    if wuji.get("state_wuji_left_20") is not None:
        print("WUJI_STATE_LEFT_20 = " + _fmt_python_list(wuji["state_wuji_left_20"]))
    if wuji.get("state_wuji_right_20") is not None:
        print("WUJI_STATE_RIGHT_20 = " + _fmt_python_list(wuji["state_wuji_right_20"]))
    if wuji.get("t_action_wuji_left") is not None or wuji.get("t_action_wuji_right") is not None:
        print(f"WUJI_T_ACTION = left:{wuji.get('t_action_wuji_left')} right:{wuji.get('t_action_wuji_right')}")
    if wuji.get("t_state_wuji_left") is not None or wuji.get("t_state_wuji_right") is not None:
        print(f"WUJI_T_STATE  = left:{wuji.get('t_state_wuji_left')} right:{wuji.get('t_state_wuji_right')}")

    print("\n--- 复制到 teleop 的建议片段（用于你改按 k 时的默认姿态）---")
    print("# 在 deploy_real/xrobot_teleop_to_robot_w_hand_keyboard.py 的 send_to_redis() 里，")
    print("# `if not send_enabled:` 分支目前用 DEFAULT_MIMIC_OBS + 手全0。你可以把它改成：\n")

    if body is not None:
        print(f"safe_mimic = {_fmt_python_list(body)}")
    else:
        print("safe_mimic = <BODY_35 未读到>")
    if hand_left is not None:
        print(f"safe_hand_left = {_fmt_python_list(hand_left)}")
    else:
        print("safe_hand_left = <HAND_LEFT_7 未读到>")
    if hand_right is not None:
        print(f"safe_hand_right = {_fmt_python_list(hand_right)}")
    else:
        print("safe_hand_right = <HAND_RIGHT_7 未读到>")
    if neck is not None:
        print(f"safe_neck = {_fmt_python_list(neck)}")
    else:
        print("safe_neck = <NECK_2 未读到>")

    print("\n# 然后写回 Redis：")
    print('self.redis_pipeline.set("action_body_unitree_g1_with_hands", json.dumps(safe_mimic))')
    print('self.redis_pipeline.set("action_hand_left_unitree_g1_with_hands", json.dumps(safe_hand_left))')
    print('self.redis_pipeline.set("action_hand_right_unitree_g1_with_hands", json.dumps(safe_hand_right))')
    print('self.redis_pipeline.set("action_neck_unitree_g1_with_hands", json.dumps(safe_neck))')

    print("\n--- Wuji（通过 hand_tracking 实现 k 回默认 / p 保持）的建议 ---")
    print("# 当前实现推荐：teleop 写 wuji_hand_mode_*：")
    print("# - k => default：Wuji server 用 zero_pose 回零位（不需要 hand_tracking）")
    print("# - p => hold：Wuji server 重复 last_qpos 保持（不需要 hand_tracking）")
    print("# - 正常 => follow：Wuji server 读取 hand_tracking 并 retarget")
    if hand_tracking_left is not None:
        print("\n# 你可以把下面两段作为 k 的默认 tracking（示例）：")
        print("default_hand_tracking_left = " + _fmt_json(hand_tracking_left))
    if hand_tracking_right is not None:
        print("default_hand_tracking_right = " + _fmt_json(hand_tracking_right))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


