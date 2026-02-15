#!/usr/bin/env bash
set -euo pipefail

# 用法示例：
#   bash xdmocap_teleop_body.sh --mocap_ip 192.168.31.134 --mocap_port 7000 --world_space geo --fps 60 --offset_to_ground
#
# 说明：
# - 只遥操全身：写 action_body_*（35D）
# - 不控制：Wuji/手/脖子（全部固定默认）

source ~/miniconda3/bin/activate gmr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

python xdmocap_teleop_body_to_redis.py "$@"




