#!/bin/bash
set -euo pipefail

# Wuji Hand Controller via Redis
# 从 Redis 读取 teleop.sh 发送的手部控制数据，实时控制 Wuji 灵巧手

source ~/miniconda3/bin/activate twist2
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "${SCRIPT_DIR}/deploy_real"

# 重要：当 hand_side=both 时我们会把两个 python 放到后台 (&)。
# 这时终端按 Ctrl+C（SIGINT）默认只会打断前台的 bash，而不会自动把后台子进程干净退出，
# 容易导致 USB 设备仍被占用（下一次启动报 ERROR_BUSY）。
# 所以这里用 trap 做“强制回收”：收到 INT/TERM/EXIT 时 kill 子进程并 wait。
PIDS=()
cleanup_children() {
  # 避免递归触发
  trap - INT TERM EXIT
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo ""
    echo "[local] 🛑 捕获退出信号，正在停止 ${#PIDS[@]} 个 Wuji 控制进程..."
    for pid in "${PIDS[@]}"; do
      kill -TERM "${pid}" 2>/dev/null || true
    done
    # 给一点时间让 python 跑 cleanup（释放 USB）
    sleep 0.2 || true
    for pid in "${PIDS[@]}"; do
      kill -KILL "${pid}" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
      wait "${pid}" 2>/dev/null || true
    done
  fi
}
trap cleanup_children INT TERM EXIT

# 配置参数（可通过命令行覆盖）
redis_ip="${REDIS_IP:-localhost}"
hand_side="${HAND_SIDE:-left}"  # left|right|both
target_fps="${TARGET_FPS:-50}"
no_smooth="${NO_SMOOTH:-1}"
smooth_steps="${SMOOTH_STEPS:-5}"

# 多设备筛选（可选）
serial_number="${SERIAL_NUMBER:-3473384E3433}"
left_serial="${LEFT_SERIAL:-3473384E3433}"
right_serial="${RIGHT_SERIAL:-3478385B3433}"

usage() {
  cat <<EOF
用法：
  $0 [--redis_ip <ip>] [--hand_side left|right|both] [--target_fps 50]
     [--no_smooth | --smooth_steps 5]
     [--serial_number <sn>] [--left_serial <sn>] [--right_serial <sn>]

说明：
- 多台 Wuji 手同时连接时，建议指定 serial_number（或分别指定 left/right）。
- hand_side=both 时会同时启动左右手两个进程。

环境变量也可用（优先级低于命令行）：
  REDIS_IP, HAND_SIDE, TARGET_FPS, NO_SMOOTH, SMOOTH_STEPS, SERIAL_NUMBER, LEFT_SERIAL, RIGHT_SERIAL
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --redis_ip) redis_ip="$2"; shift 2;;
    --hand_side) hand_side="$2"; shift 2;;
    --target_fps) target_fps="$2"; shift 2;;
    --no_smooth) no_smooth="1"; shift 1;;
    --smooth_steps) no_smooth="0"; smooth_steps="$2"; shift 2;;
    --serial_number) serial_number="$2"; shift 2;;
    --left_serial) left_serial="$2"; shift 2;;
    --right_serial) right_serial="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1"; usage; exit 1;;
  esac
done

hand_side="$(echo "${hand_side}" | tr '[:upper:]' '[:lower:]')"
if [[ "${hand_side}" != "left" && "${hand_side}" != "right" && "${hand_side}" != "both" ]]; then
  echo "❌ --hand_side 必须是 left|right|both，当前: ${hand_side}"
  exit 1
fi

PY_COMMON=(--redis_ip "${redis_ip}" --target_fps "${target_fps}")
if [[ "${no_smooth}" == "1" ]]; then
  PY_COMMON+=(--no_smooth)
else
  PY_COMMON+=(--smooth_steps "${smooth_steps}")
fi

build_py_args_for_side() {
  local side="$1"
  local sn="${2:-}"
  local args=(--hand_side "${side}")
  args+=("${PY_COMMON[@]}")
  if [[ -n "${sn}" ]]; then
    args+=(--serial_number "${sn}")
  fi
  echo "${args[@]}"
}

if [[ "${hand_side}" == "both" ]]; then
  # If user only passed --serial_number, apply to both (convenient but less common)
  if [[ -n "${serial_number}" ]]; then
    [[ -z "${left_serial}" ]] && left_serial="${serial_number}"
    [[ -z "${right_serial}" ]] && right_serial="${serial_number}"
  fi

  echo "[local] 启动 Wuji 左右手：redis_ip=${redis_ip}, fps=${target_fps}"
  echo "[local] left_serial=${left_serial:-<auto>}, right_serial=${right_serial:-<auto>}"
  # 单进程双手（推荐）：降低并发抖动/USB 竞争风险
  DUAL_ARGS=(--redis_ip "${redis_ip}" --target_fps "${target_fps}")
  if [[ "${no_smooth}" == "1" ]]; then
    DUAL_ARGS+=(--no_smooth)
  else
    DUAL_ARGS+=(--smooth_steps "${smooth_steps}")
  fi
  DUAL_ARGS+=(--left_serial "${left_serial}" --right_serial "${right_serial}")
  python server_wuji_hands_redis_dual.py "${DUAL_ARGS[@]}"
else
  sn="${serial_number}"
  if [[ -z "${sn}" ]]; then
    [[ "${hand_side}" == "left" ]] && sn="${left_serial}" || sn="${right_serial}"
  fi

  echo "[local] 启动 Wuji ${hand_side} 手：redis_ip=${redis_ip}, fps=${target_fps}, serial=${sn:-<auto>}"
  python server_wuji_hand_redis.py $(build_py_args_for_side "${hand_side}" "${sn}")
fi



