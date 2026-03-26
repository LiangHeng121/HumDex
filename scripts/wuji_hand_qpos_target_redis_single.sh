SCRIPT_DIR=$(dirname $(realpath $0))
cd "${SCRIPT_DIR}/../deploy_real"

redis_ip="localhost"
hand_side="left"
robot_key="unitree_g1_with_hands"
target_fps=50

python server_wuji_hand_qpos_target_redis.py \
  --hand_side "${hand_side}" \
  --robot_key "${robot_key}" \
  --redis_ip "${redis_ip}" \
  --target_fps "${target_fps}" \
  --serial_number  3555374E3533 \
  --no_smooth


