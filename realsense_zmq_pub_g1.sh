#!/bin/bash
set -euo pipefail

# 在本机执行：ssh 到 g1，在 g1 上启动 RealSense -> ZMQ PUB 图像服务器（兼容 VisionClient）
#
# 用法示例：
#   ./realsense_zmq_pub_g1.sh
#   ./realsense_zmq_pub_g1.sh --host g1 --remote_dir ~/TWIST2 --port 5555 --width 640 --height 480 --fps 30
#
# 依赖：
# - g1 上有本仓库（至少包含 deploy_real/server_realsense_zmq_pub.py）
# - g1 的 conda 环境里有：pyrealsense2, opencv-python, pyzmq

# Add this to the remote command execution string
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

HOST="${HOST:-g1}"
REMOTE_DIR="${REMOTE_DIR:-~/TWIST2}"
CONDA_ENV="${CONDA_ENV:-twist2}"

BIND="${BIND:-0.0.0.0}"
PORT="${PORT:-5555}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
JPEG_QUALITY="${JPEG_QUALITY:-80}"
RS_SERIAL="${RS_SERIAL:-}"

usage() {
  cat <<EOF
用法：
  $0 [--host g1] [--remote_dir ~/TWIST2] [--conda_env twist2]
     [--bind 0.0.0.0] [--port 5555] [--width 640] [--height 480] [--fps 30]
     [--jpeg_quality 80] [--rs_serial <serial>]

环境变量也可用（优先级低于命令行）：
  HOST, REMOTE_DIR, CONDA_ENV, BIND, PORT, WIDTH, HEIGHT, FPS, JPEG_QUALITY, RS_SERIAL
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --remote_dir) REMOTE_DIR="$2"; shift 2;;
    --conda_env) CONDA_ENV="$2"; shift 2;;
    --bind) BIND="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --width) WIDTH="$2"; shift 2;;
    --height) HEIGHT="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    --jpeg_quality) JPEG_QUALITY="$2"; shift 2;;
    --rs_serial) RS_SERIAL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1"; usage; exit 1;;
  esac
done

PY_ARGS=(--bind "${BIND}" --port "${PORT}" --width "${WIDTH}" --height "${HEIGHT}" --fps "${FPS}" --jpeg_quality "${JPEG_QUALITY}")
if [[ -n "${RS_SERIAL}" ]]; then
  PY_ARGS+=(--rs_serial "${RS_SERIAL}")
fi

REMOTE_CMD=$(cat <<'EOF'
set -euo pipefail

pick_repo_dir() {
  local d
  for d in "$1" \
           "$HOME/TWIST2" \
           "$HOME/heng/G1/TWIST2" \
           "$HOME/G1/TWIST2" \
           "$HOME/projects/TWIST2"
  do
    if [[ -f "${d}/deploy_real/server_realsense_zmq_pub.py" ]]; then
      echo "${d}"
      return 0
    fi
  done
  return 1
}

source ~/miniconda3/bin/activate "__CONDA_ENV__"

REPO_DIR="$(pick_repo_dir "__REMOTE_DIR__")" || {
  echo "❌ 在 g1 上找不到 TWIST2 仓库目录（需要包含 deploy_real/server_realsense_zmq_pub.py）"
  echo "   你传入的 --remote_dir 是: __REMOTE_DIR__"
  echo "   我尝试过的候选目录包括: __REMOTE_DIR__, ~/TWIST2, ~/heng/G1/TWIST2, ~/G1/TWIST2, ~/projects/TWIST2"
  echo "   解决方法："
  echo "     1) 先把 TWIST2 传到 g1（rsync/scp/tar 都行）"
  echo "     2) 或者用正确路径覆盖：./realsense_zmq_pub_g1.sh --remote_dir <g1上的TWIST2路径>"
  exit 2
}

cd "${REPO_DIR}/deploy_real"

# # 尝试释放 RealSense 设备占用（best-effort，不影响后续启动）
# # 说明：如果 g1 上该进程不存在或 sudo 需要密码，这里会静默跳过
# if command -v pgrep >/dev/null 2>&1 && pgrep -x videohub_pc4 >/dev/null 2>&1; then
#   echo "[g1] 检测到 videohub_pc4 正在运行，尝试停止以释放 RealSense(/dev/video*)..."
#   sudo -n killall -9 videohub_pc4 2>/dev/null || killall -9 videohub_pc4 2>/dev/null || true
#   sleep 0.2
#   if pgrep -x videohub_pc4 >/dev/null 2>&1; then
#     echo "❌ videohub_pc4 仍在运行（通常是 root 进程，需要 sudo 密码才能结束），因此 RealSense 仍会 busy。"
#     echo "   请你在 g1 上手动执行以下任一命令（需要输入 sudo 密码）："
#     echo "     sudo killall -9 videohub_pc4"
#     echo "     或者：sudo pkill -9 videohub_pc4"
#     echo "   结束后再重新运行本脚本。"
#     exit 3
#   fi
# fi

sudo killall -9 videohub_pc4 >/dev/null 2>&1 || true
sleep 0.1

exec /usr/bin/python3 server_realsense_zmq_pub.py __PY_ARGS__
EOF
)

REMOTE_CMD="${REMOTE_CMD/__CONDA_ENV__/${CONDA_ENV}}"
REMOTE_CMD="${REMOTE_CMD/__REMOTE_DIR__/${REMOTE_DIR}}"
REMOTE_CMD="${REMOTE_CMD/__PY_ARGS__/${PY_ARGS[*]}}"

echo "[local] ssh ${HOST} 启动 RealSense ZMQ PUB..."
ssh -t "${HOST}" "bash -lc $(printf '%q' "${REMOTE_CMD}")"


