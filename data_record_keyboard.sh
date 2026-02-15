#!/bin/bash
set -euo pipefail

# 键盘录制版数据采集：
# - 图像：默认走 ZMQ（兼容 ZED / RealSense 只要发布端是同协议 JPEG ZMQ PUB）
# - 数据：从 Redis 同时读取身体 state/action + 手的 hand_tracking_left/right
#
# 键盘控制：
# - r：开始/停止（停止会触发保存）
# - q：退出

# sudo usermod -aG input heng
# newgrp input

USER_HOME="${HOME}"
if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
  # sudo 后 HOME 会变成 /root，导致找不到原用户的 conda
  USER_HOME="$(eval echo "~${SUDO_USER}")"
fi

CONDA_ACTIVATE="${USER_HOME}/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" twist2
else
  echo "❌ 找不到 conda activate: ${CONDA_ACTIVATE}"
  echo "   解决：确认 miniconda3 路径，或直接用 env python 运行：${USER_HOME}/miniconda3/envs/twist2/bin/python ..."
  exit 1
fi

cd deploy_real

# Redis（teleop/sim2real 写入的那个 Redis）
redis_ip="localhost"

# 图像服务器（在 g1 上跑的相机发布端 IP/端口）
vision_ip="192.168.123.164"
vision_port=5555

data_frequency=30
task_name=$(date +"%Y%m%d_%H%M")

python server_data_record_keyboard_realsense.py \
  --redis_ip "${redis_ip}" \
  --frequency "${data_frequency}" \
  --task_name "${task_name}" \
  --vision_backend zmq \
  --vision_ip "${vision_ip}" \
  --vision_port "${vision_port}" \
  --save_episode_video \
  --keyboard_backend evdev \
  --evdev_device /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd \
  "$@"


