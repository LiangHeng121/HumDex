#!/bin/bash

# TWIST2 Real Robot Controller with Wuji Hand
# 控制 G1 机器人身体 + Wuji 灵巧手

source ~/miniconda3/bin/activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# change the network interface name to your own that connects to the robot
# net=enp0s31f6
# net=eno1
net=enp12s0  # 连接机器人的网口

cd deploy_real

python server_low_level_g1_real_wuji_hand.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cpu \
    --use_wuji_hand \
    --wuji_hand_sides left \
    --wuji_hand_smooth \
    --wuji_hand_smooth_steps 5
    # --smooth_body 0.5
    # --record_proprio \

