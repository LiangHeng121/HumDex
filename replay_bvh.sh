#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash replay_bvh.sh --bvh_file <path.bvh> --format nokov --fps 30 --loop
#
# 说明：
# - 与 teleop.sh 一致：使用 conda env `gmr`
# - 只回放全身（写 action_body_*），不控制 Wuji/手/脖子

source ~/miniconda3/bin/activate gmr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

python replay_bvh_body_to_redis.py "$@"


