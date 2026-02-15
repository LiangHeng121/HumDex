#!/bin/bash
set -e

# PICO retarget model -> Wuji hand MuJoCo sim (via Redis)
#
# 前置条件：
# - 你已经在另一个终端跑了 PICO 连接/发布（参考 teleop.sh）
#   它会持续写入 Redis key: hand_tracking_left/right_unitree_g1_with_hands
#
# 用法：
#   bash pico_wuji_hand_model_sim.sh \
#     --hand_side left \
#     --policy_tag wuji_left_pico --policy_epoch 50
#
#   bash pico_wuji_hand_model_sim.sh \
#     --hand_side both \
#     --policy_tag_left wuji_left_pico --policy_epoch_left 50 \
#     --policy_tag_right wuji_right_pico --policy_epoch_right 50

source ~/miniconda3/bin/activate retarget

SCRIPT_DIR=$(dirname $(realpath $0))
cd "${SCRIPT_DIR}"

cd wuji_retargeting/example
python pico_retarget_sim.py "$@"

