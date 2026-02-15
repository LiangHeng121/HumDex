#!/usr/bin/env bash
set -euo pipefail

# 3D Pose CSV Replay (BODY ONLY) -> Redis
# 用途：把 XDMocap 的 FK CSV（Geo 世界系、全局 quat）转到 GMR 期望的世界系，并做全身重定向后写入 Redis。
#
# 当前默认链路（你已经验证 OK）：
#   CSV(Geo world) --(官方 CoordinateGeo2Bvh / QuatGeo2Bvh)--> BVH raw world
#               --(BVH->GMR 固定轴旋转)--> GMR world
#               --(GMR IK retarget)--> action_body_* (35D) -> Redis
#
# 说明：
# - 脚本会自动 activate conda 环境 gmr，并 cd 到 deploy_real
# - 只控制全身：写 action_body_*（35D）
# - 不控制：Wuji/手/脖子（全部固定默认）

source ~/miniconda3/bin/activate gmr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

# 直接写死常用启动参数（你只需要：bash replay_3d_bvh.sh）
# 额外参数可直接追加：bash replay_3d_bvh.sh --start 200 --end 600
python replay_3d_bvh_body_to_redis.py \
  --csv_file /home/heng/heng/G1/TWIST2/xdmocap/data/motionData_20260108210128.csv \
  --format nokov \
  --redis_ip localhost \
  --csv_geo_to_bvh_official \
  --csv_apply_bvh_rotation \
  --use_csv_fps \
  --offset_to_ground \
  "$@"

# 备注：
# - XDMocap SDK（WS_Geo）输出：position=xyz(m)，quaternion=wxyz(全局)
# - --csv_geo_to_bvh_official 使用官方映射：pos(-x,z,y), quat(w,-x,z,y)
# - --csv_apply_bvh_rotation 将 BVH raw world 再转到 GMR 约定世界系（与 BVH loader 对齐）
