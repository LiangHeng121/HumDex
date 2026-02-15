#!/bin/bash
set -euo pipefail

# 删除 TWIST2 流程相关 Redis keys（避免录制读到旧值）
#
# 默认：dry-run（只打印，不删除）
# 加 --apply：执行删除
#
# 示例：
#   bash clear_twist2_redis_keys.sh --redis_ip 127.0.0.1 --robot_key unitree_g1_with_hands --dry_run
#   bash clear_twist2_redis_keys.sh --redis_ip 127.0.0.1 --robot_key unitree_g1_with_hands --apply
#
# 可选：
#   --delete_controller_data  : 额外删除旧版录制用的 controller_data

REDIS_IP="${REDIS_IP:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
ROBOT_KEY="${ROBOT_KEY:-unitree_g1_with_hands}"

APPLY="0"
DELETE_CONTROLLER_DATA="0"

usage() {
  cat <<EOF
用法：
  $0 --redis_ip <ip> [--redis_port 6379] [--robot_key unitree_g1_with_hands] [--dry_run|--apply]
     [--delete_controller_data]

环境变量（优先级低于命令行）：
  REDIS_IP, REDIS_PORT, ROBOT_KEY
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --redis_ip) REDIS_IP="$2"; shift 2;;
    --redis_port) REDIS_PORT="$2"; shift 2;;
    --robot_key) ROBOT_KEY="$2"; shift 2;;
    --dry_run) APPLY="0"; shift 1;;
    --apply) APPLY="1"; shift 1;;
    --delete_controller_data) DELETE_CONTROLLER_DATA="1"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1"; usage; exit 1;;
  esac
done

suffix="${ROBOT_KEY}"
keys=(
  # state/action（body/hand/neck）
  "state_body_${suffix}"
  "state_hand_left_${suffix}"
  "state_hand_right_${suffix}"
  "state_neck_${suffix}"
  "t_state"
  "action_body_${suffix}"
  "action_hand_left_${suffix}"
  "action_hand_right_${suffix}"
  "action_neck_${suffix}"
  "t_action"

  # teleop hand tracking（Wuji 输入）
  "hand_tracking_left_${suffix}"
  "hand_tracking_right_${suffix}"

  # Wuji 手写回（可选，但建议一起清掉，避免读旧值）
  "action_wuji_qpos_target_left_${suffix}"
  "action_wuji_qpos_target_right_${suffix}"
  "t_action_wuji_hand_left_${suffix}"
  "t_action_wuji_hand_right_${suffix}"
  "state_wuji_hand_left_${suffix}"
  "state_wuji_hand_right_${suffix}"
  "t_state_wuji_hand_left_${suffix}"
  "t_state_wuji_hand_right_${suffix}"
)

if [[ "${DELETE_CONTROLLER_DATA}" == "1" ]]; then
  keys+=("controller_data")
fi

echo "[info] Redis: ${REDIS_IP}:${REDIS_PORT}"
echo "[info] robot_key: ${ROBOT_KEY}"
echo "[info] mode: $([[ "${APPLY}" == "1" ]] && echo APPLY || echo DRY_RUN)"
echo ""
echo "[keys] 将处理以下 keys："
for k in "${keys[@]}"; do
  echo "  - ${k}"
done
echo ""

if ! command -v redis-cli >/dev/null 2>&1; then
  echo "❌ 未找到 redis-cli。请安装 redis-tools，或在有 redis-cli 的机器上运行。"
  exit 2
fi

if [[ "${APPLY}" != "1" ]]; then
  echo "[dry-run] 未执行删除。要真正删除请加 --apply"
  exit 0
fi

# 执行删除
deleted=$(
  redis-cli -h "${REDIS_IP}" -p "${REDIS_PORT}" DEL "${keys[@]}" 2>/dev/null || true
)
echo "[done] DEL 返回：${deleted}（删除的 key 数量）"


