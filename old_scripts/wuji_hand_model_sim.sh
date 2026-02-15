#!/bin/bash

# Wuji Hand Simulation via Redis (MuJoCo)
# 从 Redis 读取 wuji 20D action（deploy2.py 写入的 action_wuji_qpos_target_*），并在 wuji_retargeting 的 MuJoCo 仿真中执行。

set -e

source ~/miniconda3/bin/activate retarget

SCRIPT_DIR=$(dirname $(realpath $0))
cd "${SCRIPT_DIR}"

# 配置参数
redis_ip="localhost"
redis_port=6379
hand_side="right"   # "left" or "right"
target_fps=50
robot_key="unitree_g1_with_hands"

# 对齐 deploy2.py 的关键参数
policy_tag="wuji_right_mse_pico_2"
policy_epoch=300

clamp_min=-1.5
clamp_max=1.5
max_delta_per_step=0.08

cd wuji_retargeting/example

python teleop_sim_redis.py \
  --hand_side "${hand_side}" \
  --redis_ip "${redis_ip}" \
  --redis_port "${redis_port}" \
  --robot_key "${robot_key}" \
  --target_fps "${target_fps}" \
  --policy_tag "${policy_tag}" \
  --policy_epoch "${policy_epoch}" \
  --clamp_min "${clamp_min}" \
  --clamp_max "${clamp_max}" \
  --max_delta_per_step "${max_delta_per_step}"


