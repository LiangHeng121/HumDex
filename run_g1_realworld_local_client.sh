SCRIPT_DIR=$(dirname $(realpath $0))
cd "${SCRIPT_DIR}"

server_host="127.0.0.1"
server_port=8000
redis_ip="localhost"
robot_key="unitree_g1_with_hands"
vision_host="192.168.123.164"
vision_port=5555
obs_frames=4
control_fps=30
num_chunks=100
prompt="pick up bread"
debug_dir="../results_g1_real_world/debug"

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python -m deploy_real.g1_local_exec_client \
  --server_host "${server_host}" \
  --server_port "${server_port}" \
  --redis_ip "${redis_ip}" \
  --robot_key "${robot_key}" \
  --vision_host "${vision_host}" \
  --vision_port "${vision_port}" \
  --obs_frames "${obs_frames}" \
  --control_fps "${control_fps}" \
  --num_chunks "${num_chunks}" \
  --prompt "${prompt}" \
  --debug_dir "${debug_dir}"

