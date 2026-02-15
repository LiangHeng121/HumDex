#!/usr/bin/env bash
set -euo pipefail

# 3D Pose CSV Replay (WUJI ONLY) -> Redis
# 用途：把 XDMocap 的 FK CSV（Geo 世界系、全局 quat）转到 GMR 期望的世界系，并把手部 tracking 写入 Redis（Wuji follow）。
#
# 默认链路（与 replay_3d_bvh.sh 一致）：
#   CSV(Geo world) --(官方 CoordinateGeo2Bvh / QuatGeo2Bvh)--> BVH raw world
#               --(BVH->GMR 固定轴旋转)--> GMR world
#               --(提取手部关键点)--> hand_tracking_* -> Redis
#
# 说明：
# - 脚本会自动 activate conda 环境 gmr，并 cd 到 deploy_real
# - 只控制 Wuji：写 hand_tracking_left/right_* 且把 wuji_hand_mode_* 置为 follow
# - 不写 action_body/action_neck/action_hand_*（不控制全身/脖子/Unitree 夹爪手）

source ~/miniconda3/bin/activate gmr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

# 直接写死常用启动参数（你只需要：bash replay_3d_bvh_wuji.sh）
# 额外参数可直接追加：bash replay_3d_bvh_wuji.sh --start 200 --end 600 --viz
python replay_3d_bvh_wuji_to_redis.py \
  --csv_file /home/heng/heng/G1/TWIST2/xdmocap/data/20260108_130210_chunk000_pose.csv \
  --redis_ip localhost \
  --csv_geo_to_bvh_official \
  --csv_apply_bvh_rotation \
  --use_csv_fps \
  --hands left \
  --hand_fk \
  --hand_fk_end_site_scale 0.8 \
  --viz \
  --viz_layout left \
  --viz_coords wrist_local \
  --viz_every 2 \
  --viz_scale 100 \
  --viz_fixed_range 25 \
  "$@"

# 备注：
# - XDMocap SDK（WS_Geo）输出：position=xyz(m)，quaternion=wxyz(全局)
# - --csv_geo_to_bvh_official 使用官方映射：pos(-x,z,y), quat(w,-x,z,y)
# - --csv_apply_bvh_rotation 将 BVH raw world 再转到 GMR 约定世界系（与 BVH loader 对齐）

 
