#!/bin/bash

# Wuji Hand Controller via Redis
# 从 Redis 读取 teleop.sh 发送的手部控制数据，实时控制 Wuji 灵巧手

source ~/miniconda3/bin/activate twist2
SCRIPT_DIR=$(dirname $(realpath $0))
cd deploy_real

# 配置参数
redis_ip="localhost"
hand_side="right"  # "left" 或 "right"
# hand_side="right"  # "left" 或 "right"
target_fps=100

# 运行控制器
python server_wuji_hand_redis.py \
    --hand_side ${hand_side} \
    --serial_number 357937583533 \
    --redis_ip ${redis_ip} \
    --target_fps ${target_fps} \
    --no_smooth \
    --debug_pinch \
    --debug_pinch_every 50 \
    # --use_model \
    # --policy_tag wuji_right_mse_3 \
    # --policy_epoch 200
    # --pinch_project_ratio 0.03 \
    # --pinch_escape_ratio 0.04 \
    # --pinch_project_dist_max 0.12 \
    # --disable_dexpilot_projection


