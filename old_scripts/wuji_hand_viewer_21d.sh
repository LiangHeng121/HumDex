#!/bin/bash

# Standalone hand viewer (matplotlib 3D) from Redis hand_tracking_*.
# Does NOT control the robot/hand; safe to run alongside control loops.

source ~/miniconda3/bin/activate twist2
SCRIPT_DIR=$(dirname $(realpath $0))
cd "${SCRIPT_DIR}"

redis_ip="localhost"
hand_side="right"   # left/right
robot_key="unitree_g1_with_hands"

python deploy_real/viz_hand21d_redis.py \
  --redis_ip "${redis_ip}" \
  --hand_side "${hand_side}" \
  --robot_key "${robot_key}" \
  --mode 21 \
  --fps 20 \
  --lines


