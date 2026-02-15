#!/usr/bin/env bash
set -euo pipefail

# One-click: XDMocap teleop + (internal) sim2real + (internal) Wuji + RealSense recorder
# 不改原脚本：只提供一个新的入口，方便你“穿设备做任务”。
#
# 默认行为：
# - 使用本机直连 RealSense 录视觉（realsense backend）
# - 后台启动 xdmocap_teleop.sh
# - 录制端可选本地计算 Wuji 手 qpos_target（写回 Redis 供下游使用/录制）
# - 窗口里按 r 开始/停止录制，q 退出
#
# 手部 qpos_target 计算（本地，record-only）：
# - 默认：DexPilot retarget（与原逻辑一致）
# - 可选：GeoRT 模型推理（类似 wuji_hand_model_deploy.sh 的接口）
#
# 例子（用 GeoRT 模型推理替换本地 retarget）：
#   bash data_record_xdmocap_oneclick.sh \
#     --local_wuji_use_model 1 \
#     --local_wuji_policy_tag_left  wuji_mse_left  --local_wuji_policy_epoch_left  200 \
#     --local_wuji_policy_tag_right wuji_mse_right --local_wuji_policy_epoch_right 200

source ~/miniconda3/bin/activate twist2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

# 对齐 sim2real.sh 的默认
ckpt_path="${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx"
net="enp14s0"

python server_data_record_xdmocap_oneclick_realsense.py \
  --wuji_hands right \
  --redis_ip localhost \
  --frequency 30 \
  --device cpu \
  --rs_w 640 \
  --rs_h 480 \
  --rs_fps 30 \
  "$@"


