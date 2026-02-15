#!/usr/bin/env python3
"""
PICO retarget model -> Wuji hand MuJoCo SIM (via Redis).

This is a thin orchestrator around `wuji_retargeting/example/teleop_sim_redis.py`.

Why this file:
- `teleop_sim_redis.py` already implements the dataflow:
    Redis hand_tracking_* (26D dict) -> 21D mediapipe -> GeoRT policy -> wuji 20D -> MuJoCo
- For PICO usage we often want to:
    - run left+right together (two processes)
    - switch to a different policy_tag/epoch (your PICO-trained retarget model)

Usage examples:
  # Left hand only
  python pico_retarget_sim.py --hand_side left --policy_tag wuji_left_pico --policy_epoch 50

  # Both hands (two processes)
  python pico_retarget_sim.py --hand_side both \
    --policy_tag_left wuji_left_pico --policy_epoch_left 50 \
    --policy_tag_right wuji_right_pico --policy_epoch_right 50

Notes:
- This script assumes the PICO connection side is already publishing Redis keys:
    hand_tracking_left_<robot_key>, hand_tracking_right_<robot_key>, wuji_hand_mode_*_<robot_key>
  (See deploy_real/xrobot_teleop_to_robot_w_hand_keyboard.py and teleop.sh)
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


HERE = Path(__file__).resolve().parent
TELEOP_SIM_REDIS = HERE / "teleop_sim_redis.py"


def _build_child_cmd(
    *,
    hand_side: str,
    redis_ip: str,
    redis_port: int,
    robot_key: str,
    target_fps: int,
    stale_ms: int,
    no_smooth: bool,
    smooth_steps: int,
    policy_tag: str,
    policy_epoch: int,
    use_fingertips5: bool,
    clamp_min: float,
    clamp_max: float,
    max_delta_per_step: float,
) -> List[str]:
    cmd = [
        sys.executable,
        str(TELEOP_SIM_REDIS),
        "--hand_side",
        str(hand_side),
        "--redis_ip",
        str(redis_ip),
        "--redis_port",
        str(int(redis_port)),
        "--robot_key",
        str(robot_key),
        "--target_fps",
        str(int(target_fps)),
        "--stale_ms",
        str(int(stale_ms)),
        "--policy_tag",
        str(policy_tag),
        "--policy_epoch",
        str(int(policy_epoch)),
        "--clamp_min",
        str(float(clamp_min)),
        "--clamp_max",
        str(float(clamp_max)),
        "--max_delta_per_step",
        str(float(max_delta_per_step)),
    ]

    if no_smooth:
        cmd.append("--no_smooth")
    else:
        cmd += ["--smooth_steps", str(int(smooth_steps))]

    # teleop_sim_redis.py has default use_fingertips5=True; allow explicit disabling
    if use_fingertips5:
        cmd.append("--use_fingertips5")
    else:
        # no flag = False only if we add a new flag there, which we don't.
        # So we pass a negative by using env var understood by that script? Not available.
        # Keep behavior: when False here, still run without --use_fingertips5 so user can
        # patch teleop_sim_redis.py defaults if needed.
        pass

    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PICO retarget model -> Wuji hand MuJoCo sim (via Redis).")
    p.add_argument("--hand_side", choices=["left", "right", "both"], default="left")
    p.add_argument("--redis_ip", default="localhost")
    p.add_argument("--redis_port", type=int, default=6379)
    p.add_argument("--robot_key", default="unitree_g1_with_hands")
    p.add_argument("--target_fps", type=int, default=50)
    p.add_argument("--stale_ms", type=int, default=500)

    # Model selection
    p.add_argument("--policy_tag", default="", help="If set, use this tag for single-hand mode.")
    p.add_argument("--policy_epoch", type=int, default=-1, help="If set, use this epoch for single-hand mode.")
    p.add_argument("--policy_tag_left", default="", help="Left hand policy tag (for --hand_side both)")
    p.add_argument("--policy_epoch_left", type=int, default=-1, help="Left hand policy epoch (for --hand_side both)")
    p.add_argument("--policy_tag_right", default="", help="Right hand policy tag (for --hand_side both)")
    p.add_argument("--policy_epoch_right", type=int, default=-1, help="Right hand policy epoch (for --hand_side both)")

    p.add_argument("--use_fingertips5", action="store_true", help="Use 5 fingertips only as policy input (matches many GeoRT configs).")
    p.set_defaults(use_fingertips5=True)

    # Safety / smoothing
    p.add_argument("--no_smooth", action="store_true")
    p.add_argument("--smooth_steps", type=int, default=5)
    p.add_argument("--clamp_min", type=float, default=-1.5)
    p.add_argument("--clamp_max", type=float, default=1.5)
    p.add_argument("--max_delta_per_step", type=float, default=0.08)
    return p.parse_args()


def _resolve_single_policy(args: argparse.Namespace) -> tuple[str, int]:
    tag = str(args.policy_tag).strip()
    epoch = int(args.policy_epoch)
    if tag == "":
        # fall back to the existing default in teleop_sim_redis.py
        tag = "geort_filter_wuji"
    return tag, epoch


def main() -> int:
    if not TELEOP_SIM_REDIS.exists():
        print(f"❌ Missing file: {TELEOP_SIM_REDIS}", file=sys.stderr)
        return 2

    args = parse_args()

    procs: List[subprocess.Popen] = []

    def _stop_children():
        for p in procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass

    def _handle_signal(_signum, _frame):
        _stop_children()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        if args.hand_side in ("left", "right"):
            tag, epoch = _resolve_single_policy(args)
            cmd = _build_child_cmd(
                hand_side=str(args.hand_side),
                redis_ip=str(args.redis_ip),
                redis_port=int(args.redis_port),
                robot_key=str(args.robot_key),
                target_fps=int(args.target_fps),
                stale_ms=int(args.stale_ms),
                no_smooth=bool(args.no_smooth),
                smooth_steps=int(args.smooth_steps),
                policy_tag=tag,
                policy_epoch=int(epoch),
                use_fingertips5=bool(args.use_fingertips5),
                clamp_min=float(args.clamp_min),
                clamp_max=float(args.clamp_max),
                max_delta_per_step=float(args.max_delta_per_step),
            )
            return subprocess.call(cmd)

        # both hands
        left_tag = str(args.policy_tag_left or "").strip()
        right_tag = str(args.policy_tag_right or "").strip()
        if left_tag == "" or right_tag == "":
            print("❌ --hand_side both 需要同时提供 --policy_tag_left 和 --policy_tag_right", file=sys.stderr)
            return 2

        left_epoch = int(args.policy_epoch_left)
        right_epoch = int(args.policy_epoch_right)

        cmd_l = _build_child_cmd(
            hand_side="left",
            redis_ip=str(args.redis_ip),
            redis_port=int(args.redis_port),
            robot_key=str(args.robot_key),
            target_fps=int(args.target_fps),
            stale_ms=int(args.stale_ms),
            no_smooth=bool(args.no_smooth),
            smooth_steps=int(args.smooth_steps),
            policy_tag=left_tag,
            policy_epoch=left_epoch,
            use_fingertips5=bool(args.use_fingertips5),
            clamp_min=float(args.clamp_min),
            clamp_max=float(args.clamp_max),
            max_delta_per_step=float(args.max_delta_per_step),
        )
        cmd_r = _build_child_cmd(
            hand_side="right",
            redis_ip=str(args.redis_ip),
            redis_port=int(args.redis_port),
            robot_key=str(args.robot_key),
            target_fps=int(args.target_fps),
            stale_ms=int(args.stale_ms),
            no_smooth=bool(args.no_smooth),
            smooth_steps=int(args.smooth_steps),
            policy_tag=right_tag,
            policy_epoch=right_epoch,
            use_fingertips5=bool(args.use_fingertips5),
            clamp_min=float(args.clamp_min),
            clamp_max=float(args.clamp_max),
            max_delta_per_step=float(args.max_delta_per_step),
        )

        # Launch two viewers
        procs.append(subprocess.Popen(cmd_l, env=os.environ.copy()))
        procs.append(subprocess.Popen(cmd_r, env=os.environ.copy()))

        # Wait until any exits; then stop the other.
        while True:
            for p in procs:
                rc = p.poll()
                if rc is not None:
                    _stop_children()
                    return int(rc)
            signal.pause()

    except KeyboardInterrupt:
        _stop_children()
        return 130
    finally:
        _stop_children()


if __name__ == "__main__":
    raise SystemExit(main())

