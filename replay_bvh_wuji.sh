#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash replay_bvh_wuji.sh --bvh_file <path.bvh> --format nokov --fps 30 --hands both --loop
#
# 说明：
# - 与 teleop.sh 一致：使用 conda env `gmr`
# - 只回放 Wuji（写 hand_tracking_* + wuji_hand_mode_*），不写全身 action_*

source ~/miniconda3/bin/activate gmr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

python replay_bvh_wuji_to_redis.py "$@"


