#!/bin/bash
#
# Wuji Hand Controller via Redis (Both Hands)
# 同时启动左右手两个 deploy2.py 进程，从 Redis 读取 teleop 数据并下发到两只 Wuji 手。
#
# 说明：
# - deploy2.py 内部的 redis key 会按 hand_side 区分：hand_tracking_left_* / hand_tracking_right_* 等，不会互相冲突
# - 如果两只手同时连接在同一台机器上，强烈建议填写各自的 serial_number（否则 wujihandpy.Hand() 可能选到同一设备）
#
# 用法：
#   # 最常用：双手真机 + GeoRT 模型推理（tag/epoch）+ 指定两只手 serial（避免 “2 devices found”）
#   bash wuji_hand_model_deploy_both.sh \
#     --serial_left 3555374E3533 --serial_right 357937583533 \
#     --tag_left wuji_mse_left --epoch_left 200 \
#     --tag_right wuji_mse_right --epoch_right 200
#
#   # 可选：改 redis / fps / 平滑
#   bash wuji_hand_model_deploy_both.sh --redis_ip localhost --fps_left 100 --fps_right 100 --no_smooth ...
#
# 退出：
#   Ctrl+C（脚本会给两个子进程发 SIGTERM，deploy2.py 会执行回零/失能等 cleanup）

set -euo pipefail

source ~/miniconda3/bin/activate retarget

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "${SCRIPT_DIR}"

# -------------------------------------------------------------------
# 默认配置（可用命令行参数覆盖）
# -------------------------------------------------------------------
redis_ip="localhost"

# 两个手的控制频率（建议与 teleop 侧一致）
target_fps_left=100
target_fps_right=100

# 模型（默认两只手用同一套；也可以分别配置）
policy_tag_left="wuji_left_mse"
policy_epoch_left=200

policy_tag_right="wuji_right_mse"
policy_epoch_right=200

# 平滑：默认跟你单手脚本一致（--no_smooth）
no_smooth_left=true
no_smooth_right=true

# 多设备环境：建议填写两只手各自 serial_number（空字符串表示不指定）
# 例如：left_serial_number="337238793233"
left_serial_number=""
right_serial_number=""

# -------------------------------------------------------------------

print_help() {
  cat <<'EOF'
Wuji Hand Controller via Redis (Both Hands) - wuji_retarget/deploy2.py

Required (双手真机基本必填):
  --serial_left  <SERIAL>      左手 USB Serial Number
  --serial_right <SERIAL>      右手 USB Serial Number

Model (GeoRT) 选择:
  --tag_left    <TAG>          左手模型 tag（对应 geort.load_model(tag=...)）
  --epoch_left  <EPOCH>        左手模型 epoch
  --tag_right   <TAG>          右手模型 tag
  --epoch_right <EPOCH>        右手模型 epoch

Redis / FPS:
  --redis_ip  <IP>             Redis 地址（默认 localhost）
  --fps_left  <HZ>             左手控制频率（默认 100）
  --fps_right <HZ>             右手控制频率（默认 100）

Smoothing:
  --no_smooth                  两手都禁用平滑（默认）
  --smooth                     两手都启用平滑（等价于去掉 --no_smooth）

Other:
  -h, --help                   打印帮助

Example:
  bash wuji_hand_model_deploy_both.sh \
    --serial_left 3555374E3533 --serial_right 357937583533 \
    --tag_left wuji_mse_left --epoch_left 200 \
    --tag_right wuji_mse_right --epoch_right 200
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    --redis_ip)
      redis_ip="${2:-}"; shift 2;;
    --fps_left)
      target_fps_left="${2:-}"; shift 2;;
    --fps_right)
      target_fps_right="${2:-}"; shift 2;;
    --tag_left)
      policy_tag_left="${2:-}"; shift 2;;
    --epoch_left)
      policy_epoch_left="${2:-}"; shift 2;;
    --tag_right)
      policy_tag_right="${2:-}"; shift 2;;
    --epoch_right)
      policy_epoch_right="${2:-}"; shift 2;;
    --serial_left)
      left_serial_number="${2:-}"; shift 2;;
    --serial_right)
      right_serial_number="${2:-}"; shift 2;;
    --no_smooth)
      no_smooth_left=true
      no_smooth_right=true
      shift 1
      ;;
    --smooth)
      no_smooth_left=false
      no_smooth_right=false
      shift 1
      ;;
    *)
      echo "[deploy_both] Unknown argument: $1" >&2
      echo "Use --help for usage." >&2
      exit 2
      ;;
  esac
done

if [[ -z "${left_serial_number}" || -z "${right_serial_number}" ]]; then
  echo "[deploy_both] ERROR: --serial_left / --serial_right 不能为空（双手真机必须指定，否则 SDK 会报 2 devices found）。" >&2
  echo "Hint: serial 会在 wujihandpy 的报错日志里打印，例如 3555374E3533 / 357937583533" >&2
  exit 1
fi

cd wuji_retarget

LEFT_PID=""
RIGHT_PID=""

cleanup() {
  set +e
  echo
  echo "[deploy_both] stopping..."

  # 优先发 SIGTERM 让 deploy2.py 自己做 cleanup（回零/失能）
  if [[ -n "${LEFT_PID}" ]] && kill -0 "${LEFT_PID}" 2>/dev/null; then
    kill -TERM "${LEFT_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RIGHT_PID}" ]] && kill -0 "${RIGHT_PID}" 2>/dev/null; then
    kill -TERM "${RIGHT_PID}" 2>/dev/null || true
  fi

  # 等待子进程退出
  if [[ -n "${LEFT_PID}" ]]; then
    wait "${LEFT_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RIGHT_PID}" ]]; then
    wait "${RIGHT_PID}" 2>/dev/null || true
  fi

  echo "[deploy_both] done."
}

trap cleanup INT TERM EXIT

echo "============================================================"
echo "Wuji Hand Controller via Redis (Both Hands)"
echo "============================================================"
echo "Redis: ${redis_ip}"
echo "LEFT : fps=${target_fps_left},  tag=${policy_tag_left}@${policy_epoch_left},  no_smooth=${no_smooth_left},  serial=${left_serial_number}"
echo "RIGHT: fps=${target_fps_right}, tag=${policy_tag_right}@${policy_epoch_right}, no_smooth=${no_smooth_right}, serial=${right_serial_number}"
echo "============================================================"

# 左手
left_args=(
  --hand_side left
  --redis_ip "${redis_ip}"
  --target_fps "${target_fps_left}"
  --policy_tag "${policy_tag_left}"
  --policy_epoch "${policy_epoch_left}"
)
if [[ "${no_smooth_left}" == "true" ]]; then
  left_args+=( --no_smooth )
fi
if [[ -n "${left_serial_number}" ]]; then
  left_args+=( --serial_number "${left_serial_number}" )
fi

# 右手
right_args=(
  --hand_side right
  --redis_ip "${redis_ip}"
  --target_fps "${target_fps_right}"
  --policy_tag "${policy_tag_right}"
  --policy_epoch "${policy_epoch_right}"
)
if [[ "${no_smooth_right}" == "true" ]]; then
  right_args+=( --no_smooth )
fi
if [[ -n "${right_serial_number}" ]]; then
  right_args+=( --serial_number "${right_serial_number}" )
fi

echo "[deploy_both] starting LEFT..."
python deploy2.py "${left_args[@]}" &
LEFT_PID=$!

sleep 0.5

echo "[deploy_both] starting RIGHT..."
python deploy2.py "${right_args[@]}" &
RIGHT_PID=$!

echo "[deploy_both] running: LEFT_PID=${LEFT_PID}, RIGHT_PID=${RIGHT_PID}"
echo "[deploy_both] press Ctrl+C to stop."

# 等待任意一个退出；如果其中一个先挂了，另一个也会被 cleanup 干掉（避免半边还在跑）
wait -n "${LEFT_PID}" "${RIGHT_PID}"


