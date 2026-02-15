#!/bin/bash
#
# Wuji Hand Retargeting (DexPilot) -> MuJoCo SIM via Redis
#
# Reads:  hand_tracking_{left/right}_unitree_g1_with_hands   (26D dict from xrobot_teleop_to_robot_w_hand.py / xdmocap_teleop_body_to_redis.py)
# Runs:   26D -> 21D (MediaPipe) -> apply_mediapipe_transformations -> WujiHandRetargeter (wuji_retargeting)
# Drives: MuJoCo model from wuji_retargeting/example/utils/mujoco-sim/model/{left,right}.xml
#
set -e

# Choose your env (must have: pinocchio, mujoco, redis)
# source ~/miniconda3/bin/activate twist2
source ~/miniconda3/bin/activate retarget

SCRIPT_DIR=$(dirname $(realpath $0))
cd "${SCRIPT_DIR}"

redis_ip="localhost"
hand_side="right"   # left | right
target_fps=60

python deploy_real/server_wuji_hand_sim_redis.py \
  --hand_side "${hand_side}" \
  --redis_ip "${redis_ip}" \
  --target_fps "${target_fps}" \
  --verbose


