#!/bin/bash
#
# Wuji Hand Controller via Redis (Both Hands, Real Hardware)
# 参考：wuji_hand_redis_single.sh + wuji_hand_model_deploy_both.sh
#
# - 使用 deploy_real/server_wuji_hand_redis.py（DexPilot retargeter）控制真机
# - 左右手各自指定 serial_number，避免 “2 devices found ... please specify Serial Number”
# - Ctrl+C 一键退出，脚本会给两个子进程发 SIGTERM
#
# 用法：
#   bash wuji_hand_redis_both.sh
#

set -euo pipefail

# 跟单手脚本保持一致
source ~/miniconda3/bin/activate twist2

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "${SCRIPT_DIR}/deploy_real"

# -------------------------------------------------------------------
# 配置参数（按需修改）
# -------------------------------------------------------------------
redis_ip="localhost"

target_fps_left=50
target_fps_right=50

# 关键：两只手的 USB Serial Number（wujihandpy 日志里会打印出来）
# 你日志里提示了两个：
#   - 3555374E3533
#   - 357937583533
left_serial_number="3555374E3533"
right_serial_number="357937583533"

# 平滑：和单手脚本一致（--no_smooth）
no_smooth_left=true
no_smooth_right=true

# 可选调试项（默认关；需要时打开）
debug_pinch=false
debug_pinch_every=50
disable_dexpilot_projection=true

# -------------------------------------------------------------------

LEFT_PID=""
RIGHT_PID=""

cleanup() {
  set +e
  echo
  echo "[wuji_redis_both] stopping..."
  if [[ -n "${LEFT_PID}" ]] && kill -0 "${LEFT_PID}" 2>/dev/null; then
    kill -TERM "${LEFT_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RIGHT_PID}" ]] && kill -0 "${RIGHT_PID}" 2>/dev/null; then
    kill -TERM "${RIGHT_PID}" 2>/dev/null || true
  fi
  if [[ -n "${LEFT_PID}" ]]; then
    wait "${LEFT_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RIGHT_PID}" ]]; then
    wait "${RIGHT_PID}" 2>/dev/null || true
  fi
  echo "[wuji_redis_both] done."
}

trap cleanup INT TERM EXIT

if [[ -z "${left_serial_number}" || -z "${right_serial_number}" ]]; then
  echo "[wuji_redis_both] ERROR: left_serial_number / right_serial_number 不能为空（双手真机必须指定）。" >&2
  exit 1
fi

echo "============================================================"
echo "Wuji Hand Controller via Redis (Both Hands, Real Hardware)"
echo "============================================================"
echo "Redis: ${redis_ip}"
echo "LEFT : fps=${target_fps_left},  serial=${left_serial_number}"
echo "RIGHT: fps=${target_fps_right}, serial=${right_serial_number}"
echo "debug_pinch=${debug_pinch} every=${debug_pinch_every}  disable_dexpilot_projection=${disable_dexpilot_projection}"
echo "============================================================"

left_args=(
  --hand_side left
  --serial_number "${left_serial_number}"
  --redis_ip "${redis_ip}"
  --target_fps "${target_fps_left}"
)
if [[ "${no_smooth_left}" == "true" ]]; then
  left_args+=( --no_smooth )
fi
if [[ "${debug_pinch}" == "true" ]]; then
  left_args+=( --debug_pinch --debug_pinch_every "${debug_pinch_every}" )
fi
if [[ "${disable_dexpilot_projection}" == "true" ]]; then
  left_args+=( --disable_dexpilot_projection )
fi

right_args=(
  --hand_side right
  --serial_number "${right_serial_number}"
  --redis_ip "${redis_ip}"
  --target_fps "${target_fps_right}"
)
if [[ "${no_smooth_right}" == "true" ]]; then
  right_args+=( --no_smooth )
fi
if [[ "${debug_pinch}" == "true" ]]; then
  right_args+=( --debug_pinch --debug_pinch_every "${debug_pinch_every}" )
fi
if [[ "${disable_dexpilot_projection}" == "true" ]]; then
  right_args+=( --disable_dexpilot_projection )
fi

echo "[wuji_redis_both] starting LEFT..."
python server_wuji_hand_redis.py "${left_args[@]}" &
LEFT_PID=$!

sleep 0.5

echo "[wuji_redis_both] starting RIGHT..."
python server_wuji_hand_redis.py "${right_args[@]}" &
RIGHT_PID=$!

echo "[wuji_redis_both] running: LEFT_PID=${LEFT_PID}, RIGHT_PID=${RIGHT_PID}"
echo "[wuji_redis_both] press Ctrl+C to stop."

wait -n "${LEFT_PID}" "${RIGHT_PID}"


