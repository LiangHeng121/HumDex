#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/bin/activate gmr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

redis_ip="localhost"
actual_human_height=1.45

# safe idle 支持序列（需要 deploy_real/xrobot_teleop_to_robot_w_hand_keyboard.py 新版）：
#   - 例：--safe_idle_pose_id 1,2
#     * 按 k 进入 default：先平滑到 1，再平滑到 2
#     * 再按 k 回到跟踪：先 2->1，再 1->正常跟踪

python xrobot_teleop_to_robot_w_hand_keyboard.py \
  --robot unitree_g1 \
  --actual_human_height "${actual_human_height}" \
  --redis_ip "${redis_ip}" \
  --target_fps 100 \
  --measure_fps 1 \
  --safe_idle_pose_id 2 \
  --smooth \
  --smooth_window_size 5 \
  --keyboard_toggle_send \
  --toggle_send_key k \
  --hold_position_key p \
  --start_ramp_seconds 3.0 \
  --toggle_ramp_seconds 3.0 \
  --exit_ramp_seconds 3.0 \
  --ramp_ease cosine \
  --keyboard_backend both \
  --evdev_device /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd \
  "$@"

  # --keyboard_backend evdev \
  # --evdev_device /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd \
