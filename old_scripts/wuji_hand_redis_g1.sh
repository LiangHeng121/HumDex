#!/bin/bash
set -euo pipefail

# 在本机执行：ssh 到 g1，在 g1 上启动 Wuji Hand Redis 控制器
# 依赖：g1 上已经有本仓库（或至少有 deploy_real/server_wuji_hand_redis.py + wuji_retargeting）
# 用法示例：
#   ./wuji_hand_redis_g1.sh --hand_side right
#   ./wuji_hand_redis_g1.sh --hand_side both --left_serial 337238793233 --right_serial 337238873233
#   ./wuji_hand_redis_g1.sh --host g1 --remote_dir ~/TWIST2 --redis_ip localhost --target_fps 50 --no_smooth
#
# 重要：teleop.sh 发送的数据必须写到“同一个 Redis”里：
# - 推荐：Redis 跑在 g1 上；teleop.sh 里把 redis_ip 改成 g1 的 IP；本脚本里 --redis_ip 用 localhost

HOST="${HOST:-g1}"
REMOTE_DIR="${REMOTE_DIR:-~/TWIST2}"
CONDA_ENV="${CONDA_ENV:-twist2}"

HAND_SIDE="${HAND_SIDE:-right}"   # left|right|both
# g1 侧程序要连接的 Redis 地址：这里默认填“本机（运行 teleop/sim2real 的机器）”在机器人网络下可达的 IP
REDIS_IP="${REDIS_IP:-192.168.123.222}"
# REDIS_IP="${REDIS_IP:-172.20.10.5}"
TARGET_FPS="${TARGET_FPS:-50}"
NO_SMOOTH="${NO_SMOOTH:-1}"
SMOOTH_STEPS="${SMOOTH_STEPS:-5}"

# 可选：筛选设备（多台 Wuji 手同时连接时必须指定）
SERIAL_NUMBER="${SERIAL_NUMBER:-}"
LEFT_SERIAL="${LEFT_SERIAL:-}"
RIGHT_SERIAL="${RIGHT_SERIAL:-}"

usage() {
  cat <<EOF
用法：
  $0 [--host g1] [--remote_dir ~/TWIST2] [--conda_env twist2]
     [--hand_side left|right|both] [--redis_ip <ip>] [--target_fps 50]
     [--no_smooth | --smooth_steps 5]
     [--serial_number <sn>] [--left_serial <sn>] [--right_serial <sn>]

环境变量也可用（优先级低于命令行）：
  HOST, REMOTE_DIR, CONDA_ENV, HAND_SIDE, REDIS_IP, TARGET_FPS, NO_SMOOTH, SMOOTH_STEPS, SERIAL_NUMBER, LEFT_SERIAL, RIGHT_SERIAL
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --remote_dir) REMOTE_DIR="$2"; shift 2;;
    --conda_env) CONDA_ENV="$2"; shift 2;;
    --hand_side) HAND_SIDE="$2"; shift 2;;
    --redis_ip) REDIS_IP="$2"; shift 2;;
    --target_fps) TARGET_FPS="$2"; shift 2;;
    --no_smooth) NO_SMOOTH="1"; shift 1;;
    --smooth_steps) NO_SMOOTH="0"; SMOOTH_STEPS="$2"; shift 2;;
    --serial_number) SERIAL_NUMBER="$2"; shift 2;;
    --left_serial) LEFT_SERIAL="$2"; shift 2;;
    --right_serial) RIGHT_SERIAL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1"; usage; exit 1;;
  esac
done

HAND_SIDE="$(echo "${HAND_SIDE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${HAND_SIDE}" != "left" && "${HAND_SIDE}" != "right" && "${HAND_SIDE}" != "both" ]]; then
  echo "❌ --hand_side 必须是 left|right|both，当前: ${HAND_SIDE}"
  exit 1
fi

# Common args for python
PY_COMMON=(--redis_ip "${REDIS_IP}" --target_fps "${TARGET_FPS}")
if [[ "${NO_SMOOTH}" == "1" ]]; then
  PY_COMMON+=(--no_smooth)
else
  PY_COMMON+=(--smooth_steps "${SMOOTH_STEPS}")
fi

build_py_args_for_side() {
  local side="$1"
  shift || true
  local sn="$1"
  shift || true

  local args=(--hand_side "${side}")
  args+=("${PY_COMMON[@]}")
  if [[ -n "${sn}" ]]; then
    args+=(--serial_number "${sn}")
  fi
  echo "${args[*]}"
}

if [[ "${HAND_SIDE}" == "both" ]]; then
  # 单进程双手（推荐）：用一个 Python 同时控制左右手
  LSN="${LEFT_SERIAL}"
  RSN="${RIGHT_SERIAL}"
  # If user only passed --serial_number, apply it to both (less common, but convenient)
  if [[ -n "${SERIAL_NUMBER}" ]]; then
    [[ -z "${LSN}" ]] && LSN="${SERIAL_NUMBER}"
    [[ -z "${RSN}" ]] && RSN="${SERIAL_NUMBER}"
  fi
  if [[ -z "${LSN}" || -z "${RSN}" ]]; then
    echo "❌ 双手模式需要 --left_serial 和 --right_serial（或用 --serial_number 同时指定）"
    exit 1
  fi
  DUAL_ARGS=(--redis_ip "${REDIS_IP}" --target_fps "${TARGET_FPS}" --left_serial "${LSN}" --right_serial "${RSN}")
  if [[ "${NO_SMOOTH}" == "1" ]]; then
    DUAL_ARGS+=(--no_smooth)
  else
    DUAL_ARGS+=(--smooth_steps "${SMOOTH_STEPS}")
  fi
else
  # Single hand: prefer --serial_number; fallback to side-specific
  SN="${SERIAL_NUMBER}"
  if [[ -z "${SN}" ]]; then
    if [[ "${HAND_SIDE}" == "left" ]]; then SN="${LEFT_SERIAL}"; else SN="${RIGHT_SERIAL}"; fi
  fi
  ONE_ARGS="$(build_py_args_for_side "${HAND_SIDE}" "${SN}")"
fi

REMOTE_CMD=$(cat <<'EOF'
set -euo pipefail

# 当我们在 g1 上用 & 启动多个 python 时，需要确保退出/断连时能回收子进程。
# 否则容易残留占用 USB，导致下次启动报 ERROR_BUSY。
PIDS=()
cleanup_children() {
  trap - INT TERM EXIT
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo ""
    echo "[g1] 🛑 正在停止 ${#PIDS[@]} 个 Wuji 控制进程..."
    for pid in "${PIDS[@]}"; do
      kill -TERM "${pid}" 2>/dev/null || true
    done
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

pick_repo_dir() {
  local d
  for d in "$1" \
           "$HOME/TWIST2" \
           "$HOME/heng/G1/TWIST2" \
           "$HOME/G1/TWIST2" \
           "$HOME/projects/TWIST2"
  do
    if [[ -d "${d}/deploy_real" ]]; then
      echo "${d}"
      return 0
    fi
  done
  return 1
}

source ~/miniconda3/bin/activate "__CONDA_ENV__"

REPO_DIR="$(pick_repo_dir "__REMOTE_DIR__")" || {
  echo "❌ 在 g1 上找不到 TWIST2 仓库目录（需要包含 deploy_real/）"
  echo "   你传入的 --remote_dir 是: __REMOTE_DIR__"
  echo "   我尝试过的候选目录包括: __REMOTE_DIR__, ~/TWIST2, ~/heng/G1/TWIST2, ~/G1/TWIST2, ~/projects/TWIST2"
  echo "   解决方法："
  echo "     1) 先把 TWIST2 传到 g1（rsync/scp/tar 都行）"
  echo "     2) 或者用正确路径覆盖：./wuji_hand_redis_g1.sh --remote_dir <g1上的TWIST2路径>"
  exit 2
}

cd "${REPO_DIR}/deploy_real"
if [[ "__HAND_SIDE__" == "both" ]]; then
  echo "[g1] 启动 Wuji 双手控制器（单进程）"
  echo "[g1] dual args: __DUAL_ARGS__"
  python server_wuji_hands_redis_dual.py __DUAL_ARGS__
else
  echo "[g1] 启动 Wuji 手控制器：__HAND_SIDE__"
  echo "[g1] args: __ONE_ARGS__"
  python server_wuji_hand_redis.py __ONE_ARGS__
fi
EOF
)

# 替换占位符（避免 heredoc 里被本地变量/通配符意外展开）
# 注意：这里用 // 做“全局替换”，因为占位符在 REMOTE_CMD 中会出现多次（echo + python 命令）
REMOTE_CMD="${REMOTE_CMD//__CONDA_ENV__/${CONDA_ENV}}"
REMOTE_CMD="${REMOTE_CMD//__REMOTE_DIR__/${REMOTE_DIR}}"
REMOTE_CMD="${REMOTE_CMD//__HAND_SIDE__/${HAND_SIDE}}"
REMOTE_CMD="${REMOTE_CMD//__DUAL_ARGS__/${DUAL_ARGS[*]:-}}"
REMOTE_CMD="${REMOTE_CMD//__ONE_ARGS__/${ONE_ARGS:-}}"

echo "[local] ssh ${HOST} 启动 Wuji Hand Redis 控制器..."
ssh -t "${HOST}" "bash -lc $(printf '%q' "${REMOTE_CMD}")"


