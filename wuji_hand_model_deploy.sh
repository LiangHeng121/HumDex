#!/bin/bash

# Wuji Hand Controller via Redis
# 从 Redis 读取 teleop.sh 发送的手部控制数据，实时控制 Wuji 灵巧手

source ~/miniconda3/bin/activate retarget
SCRIPT_DIR=$(dirname $(realpath $0))
cd wuji_retarget

# 配置参数
redis_ip="localhost"
# hand_side="left"  # "left" 或 "right"
hand_side="right"  # "left" 或 "right"
target_fps=100


# policy_tag="wuji_right_mse_pico_2"
# policy_epoch=400


policy_tag="wuji_right_mse_3"
policy_epoch=200



# 运行控制器
python deploy2.py \
    --hand_side ${hand_side} \
    --redis_ip ${redis_ip} \
    --target_fps ${target_fps} \
    --no_smooth \
    --policy_tag ${policy_tag} \
    --policy_epoch ${policy_epoch} \

    



# 官方给了我全手计算代码 @xdmocap/20260107全手坐标计算（包括指尖） ，里面包括fk计算end site的代码，能不能在 @deploy_real/xdmocap_teleop_body_to_redis.py  用这个计算一下，再输入wuji，让指尖更准确


