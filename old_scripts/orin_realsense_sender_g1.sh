#!/bin/bash
set -euo pipefail

# 在本机执行：ssh 到 g1，在 g1 上运行 ~/XRoboToolkit-Orin-Video-Sender/realsense.sh
#
# 你给的 g1 侧脚本内容（示意）：
#   sudo killall -9 videohub_pc4; sleep 0.1; ./OrinVideoSender --listen <ip:port>
#
# 用法示例：
#   ./orin_realsense_sender_g1.sh
#   ./orin_realsense_sender_g1.sh --host g1 --remote_dir ~/XRoboToolkit-Orin-Video-Sender
#   ./orin_realsense_sender_g1.sh --listen 192.168.10.28:13579
#
# 说明：
# - 默认“前台运行”：ssh 会一直占用当前终端；Ctrl+C 会尽量同时停止远端 OrinVideoSender（best-effort）
# - 如果 g1 上 sudo 需要密码，脚本里的 “kill videohub_pc4” 可能无法成功（会给出提示）

HOST="${HOST:-g1}"
REMOTE_DIR="${REMOTE_DIR:-~/XRoboToolkit-Orin-Video-Sender}"

# 可选：覆盖 g1 上 realsense.sh 里写死的参数。
# 注意：如果你什么参数都不传，本脚本会默认直接执行 g1 上的 ./realsense.sh（推荐）。
LISTEN="${LISTEN:-}"

# 可选：让 OrinVideoSender 同时开 ZMQ 输出（用于数据采集，避免再起另一个 RealSense 进程）
# 二选一：--zmq 输出 H264；--zmq-raw 输出原始 BGR/BGRA
ZMQ_ENDPOINT="${ZMQ_ENDPOINT:-}"
ZMQ_RAW_ENDPOINT="${ZMQ_RAW_ENDPOINT:-}"

# 是否尝试在 g1 上 kill videohub_pc4（默认开）
KILL_VIDEOHUB="${KILL_VIDEOHUB:-1}"

usage() {
  cat <<EOF
用法：
  $0 [--host g1] [--remote_dir ~/XRoboToolkit-Orin-Video-Sender]
     [--listen <ip:port>] [--zmq <endpoint> | --zmq_raw <endpoint>] [--no_kill_videohub]

环境变量（优先级低于命令行）：
  HOST, REMOTE_DIR, LISTEN, ZMQ_ENDPOINT, ZMQ_RAW_ENDPOINT, KILL_VIDEOHUB
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --remote_dir) REMOTE_DIR="$2"; shift 2;;
    --listen) LISTEN="$2"; shift 2;;
    --zmq) ZMQ_ENDPOINT="$2"; shift 2;;
    --zmq_raw) ZMQ_RAW_ENDPOINT="$2"; shift 2;;
    --no_kill_videohub) KILL_VIDEOHUB="0"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1"; usage; exit 1;;
  esac
done

if [[ -n "${ZMQ_ENDPOINT}" && -n "${ZMQ_RAW_ENDPOINT}" ]]; then
  echo "❌ --zmq 和 --zmq_raw 不能同时使用（当前 OrinVideoSender 只开一个 ZMQ PUB）。"
  exit 1
fi

# 是否需要“覆盖参数直接启动 OrinVideoSender”
# - 只要你传了 --listen 或 --zmq/--zmq_raw 任意一个，就认为你想覆盖
USE_OVERRIDE="0"
if [[ -n "${LISTEN}" || -n "${ZMQ_ENDPOINT}" || -n "${ZMQ_RAW_ENDPOINT}" ]]; then
  USE_OVERRIDE="1"
fi

cleanup_remote_best_effort() {
  # best-effort：避免你本地 Ctrl+C 后远端进程仍残留占用设备
  # 注意：这里不强依赖 sudo，优先用 pkill（普通用户可用）
  ssh -o BatchMode=yes -o ConnectTimeout=2 "${HOST}" "bash -lc 'pkill -f OrinVideoSender 2>/dev/null || true'" >/dev/null 2>&1 || true
}
trap cleanup_remote_best_effort INT TERM EXIT

REMOTE_CMD=$(cat <<'EOF'
set -euo pipefail

pick_sender_dir() {
  local d
  for d in "$1" \
           "$HOME/XRoboToolkit-Orin-Video-Sender" \
           "$HOME/projects/XRoboToolkit-Orin-Video-Sender"
  do
    if [[ -f "${d}/realsense.sh" ]]; then
      echo "${d}"
      return 0
    fi
  done
  return 1
}

SENDER_DIR="$(pick_sender_dir "__REMOTE_DIR__")" || {
  echo "❌ 在 g1 上找不到 XRoboToolkit-Orin-Video-Sender 目录（需要包含 realsense.sh）"
  echo "   你传入的 --remote_dir 是: __REMOTE_DIR__"
  echo "   我尝试过的候选目录包括: __REMOTE_DIR__, ~/XRoboToolkit-Orin-Video-Sender, ~/projects/XRoboToolkit-Orin-Video-Sender"
  exit 2
}

cd "${SENDER_DIR}"

if [[ ! -x "./OrinVideoSender" ]]; then
  echo "❌ 在 ${SENDER_DIR} 下找不到可执行文件 ./OrinVideoSender"
  echo "   请确认你是在正确目录、并且已编译/已赋予执行权限。"
  exit 3
fi

cleanup_children() {
  trap - INT TERM EXIT
  # best-effort：停止 OrinVideoSender（不需要 sudo）
  pkill -f OrinVideoSender 2>/dev/null || true
}
trap cleanup_children INT TERM EXIT

if [[ "__KILL_VIDEOHUB__" == "1" ]]; then
  # 释放 RealSense 占用（如果 sudo 需要密码，会失败；但不阻断后续启动）
  if command -v pgrep >/dev/null 2>&1 && pgrep -x videohub_pc4 >/dev/null 2>&1; then
    echo "[g1] 检测到 videohub_pc4 正在运行，尝试结束以释放设备..."
    sudo -n killall -9 videohub_pc4 2>/dev/null || killall -9 videohub_pc4 2>/dev/null || true
    sleep 0.1 || true
    if pgrep -x videohub_pc4 >/dev/null 2>&1; then
      echo "⚠️ videohub_pc4 仍在运行（可能需要 sudo 密码）。如果后续提示 busy，请在 g1 上手动执行："
      echo "   sudo killall -9 videohub_pc4"
    fi
  fi
fi

if [[ "__USE_OVERRIDE__" == "1" ]]; then
  # 覆盖启动：直接调用 OrinVideoSender（用于临时改 listen/zmq 端口等）
  # LISTEN 为空时，给一个安全默认值，避免 --listen 缺失导致程序直接退出
  LISTEN_FINAL="__LISTEN__"
  if [[ -z "${LISTEN_FINAL}" ]]; then
    LISTEN_FINAL="0.0.0.0:13579"
  fi
  EXTRA_ARGS=()
  if [[ -n "__ZMQ_ENDPOINT__" ]]; then
    EXTRA_ARGS+=(--zmq "__ZMQ_ENDPOINT__")
  fi
  if [[ -n "__ZMQ_RAW_ENDPOINT__" ]]; then
    EXTRA_ARGS+=(--zmq-raw "__ZMQ_RAW_ENDPOINT__")
  fi
  echo "[g1] 以覆盖参数启动：./OrinVideoSender --listen ${LISTEN_FINAL} ${EXTRA_ARGS[*]}"
  ./OrinVideoSender --listen "${LISTEN_FINAL}" "${EXTRA_ARGS[@]}"
else
  echo "[g1] 执行：bash ./realsense.sh"
  bash ./realsense.sh
fi
EOF
)

# 全局替换占位符（避免出现多处未替换）
REMOTE_CMD="${REMOTE_CMD//__REMOTE_DIR__/${REMOTE_DIR}}"
REMOTE_CMD="${REMOTE_CMD//__LISTEN__/${LISTEN}}"
REMOTE_CMD="${REMOTE_CMD//__ZMQ_ENDPOINT__/${ZMQ_ENDPOINT}}"
REMOTE_CMD="${REMOTE_CMD//__ZMQ_RAW_ENDPOINT__/${ZMQ_RAW_ENDPOINT}}"
REMOTE_CMD="${REMOTE_CMD//__KILL_VIDEOHUB__/${KILL_VIDEOHUB}}"
REMOTE_CMD="${REMOTE_CMD//__USE_OVERRIDE__/${USE_OVERRIDE}}"

echo "[local] ssh ${HOST} 启动 OrinVideoSender（目录：${REMOTE_DIR}）..."
ssh -t "${HOST}" "bash -lc $(printf '%q' "${REMOTE_CMD}")"


