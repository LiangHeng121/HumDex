#!/usr/bin/env bash
set -euo pipefail

# slimevr + xdmocap 混合遥操（body 来自 slimevr VMC/OSC，手套来自 xdmocap）
# 用法示例：
#   bash slimevr_teleop.sh --vmc_port 39539 --vmc_ip 0.0.0.0 \
#                          --dst_ip_hand 192.168.31.101 --dst_port_hand 7000 --redis_ip localhost
#
# 说明：
# - 依赖 GMR + numpy + scipy + redis python 库，因此使用 conda env `gmr`（与 teleop.sh 一致）
# - body_source=vmc 需要 python-osc（pip install python-osc）
# - 写 action_body_* + hand_tracking_*（Wuji follow），不控制 action_neck/action_hand_*
# - 默认启用键盘开关：
#   - k: 切换是否发送到 Redis（禁用时发送安全 idle，并把 Wuji 置 default）
#   - p: 切换 hold（冻结 action_body；Wuji 置 hold）
# - safe idle 支持序列（需要 deploy_real/xdmocap_teleop_body_to_redis.py 新版）：
#   - 例：--safe_idle_pose_id 1,2
#     * 按 k 进入 default：先平滑到 1，再平滑到 2
#     * 再按 k 回到跟踪：先 2->1，再 1->正常跟踪


# sudo usermod -aG input heng
# newgrp input

source ~/miniconda3/bin/activate gmr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

# 直接写死常用启动参数（你只需要：bash xdmocap_teleop.sh）
# 额外参数可直接追加：bash slimevr_teleop.sh --vmc_port ... --dst_ip_hand ... --dst_port_hand ...
# 扩展：身体和手可以来自不同 sender：
#   - body(xdmocap): --dst_ip_body/--dst_port_body/--mocap_index_body
#   - hand(xdmocap): --dst_ip_hand/--dst_port_hand/--mocap_index_hand
#   - body(vmc): 使用 --body_source vmc + --vmc_ip/--vmc_port

  # --dst_ip_body 192.168.1.101 \
  # --dst_port_body 7000 \
  # --dst_ip_hand 192.168.1.101 \
  # --dst_port_hand 7000 \
  # --mocap_index_body 0 \
  # --mocap_index_hand 0 \
  # 192.168.1.119 
# 不传则默认复用 --dst_ip/--dst_port/--mocap_index（与旧行为一致）
python slimevr_teleop_body_to_redis.py \
  --body_source vmc \
  --vmc_ip 0.0.0.0 \
  --vmc_port 39539 \
  --vmc_rot_mode local \
  --vmc_no_invert_zw \
  --vmc_use_fk \
  --vmc_use_viewer_fk \
  --vmc_fk_skeleton bvh \
  --vmc_bvh_path /home/heng/heng/G1/TWIST2/session251008_1501_align_mocap.bvh \
  --vmc_bvh_scale 0.01 \
  --vmc_viewer_bone_axis_override "Hips:flip=yz;Spine:flip=yz;Spine1:flip=yz;Neck:flip=yz;Head:flip=yz;LeftUpperArm:flip=x;RightUpperArm:flip=x;LeftLowerArm:flip=x;RightLowerArm:flip=x;LeftHand:flip=x;RightHand:flip=x;LeftUpperLeg:flip=x;RightUpperLeg:flip=x;LeftLowerLeg:flip=x;RightLowerLeg:flip=x;LeftFoot:flip=x;RightFoot:flip=x" \
  --dst_ip_hand 192.168.1.10 \
  --dst_port_hand 7000 \
  --mocap_index_body 0 \
  --mocap_index_hand 0 \
  --redis_ip localhost \
  --actual_human_height 1.45 \
  --offset_to_ground \
  --target_fps 100 \
  --measure_fps 1 \
  --hands both \
  --keyboard_toggle_send \
  --toggle_send_key k \
  --hold_position_key p \
  --hand_fk \
  --hand_fk_end_site_scale 0.858,0.882,0.882,0.882,0.882 \
  --publish_bvh_hand \
  --hand_no_csv_transform \
  --start_ramp_seconds 3.0 \
  --toggle_ramp_seconds 3.0 \
  --exit_ramp_seconds 3.0 \
  --ramp_ease cosine \
  --keyboard_backend evdev \
  --evdev_device /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd \
  --safe_idle_pose_id 2 \
  --csv_apply_bvh_rotation \
  --viz \
  "$@" \
  # 全局旋转微调（在 csv_apply_bvh_rotation 之后，按需传参覆盖）：
  # --csv_global_yaw_deg 180 \
  # --csv_global_pitch_deg 180 \
  # --csv_global_roll_deg 180 \
  # --csv_global_rot_mode world \
  # --debug_viz_stages \
  # --debug_viz_every 5 \
  # --debug_viz_viewer_axes \
  # --debug_viz_only_raw \
  # "$@" \
  # --csv_apply_bvh_rotation \
  # --csv_geo_to_bvh_official \

  # --wuji_hand_sim_viz \
  # --wuji_hand_sim_sides right \
  # --wuji_hand_sim_target_fps 60 \
  # --viz
  # "$@"
  # --wuji_hand_sim_viz \
  # --wuji_hand_sim_sides both \
  # --wuji_hand_sim_target_fps 60 \
  # 选择 sim 端用模型推理还是原 retarget：
  # --wuji_hand_sim_use_model \
  # --wuji_hand_sim_policy_tag_left  wuji_mse_left  --wuji_hand_sim_policy_epoch_left  200 \
  # --wuji_hand_sim_policy_tag_right wuji_mse_right --wuji_hand_sim_policy_epoch_right 200 \
  # --viz \

# 0.858,0.882,0.882,0.882,0.882

