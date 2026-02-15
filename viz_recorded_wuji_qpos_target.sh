#!/usr/bin/env bash
set -euo pipefail

# Quick viewer for recorded action_wuji_qpos_target_{left/right} (20D) in episodes.
# Usage:
#   bash viz_recorded_wuji_qpos_target.sh /path/to/deploy_real/twist2_demonstration/20260115_0905 --hand_side right
#   bash viz_recorded_wuji_qpos_target.sh /path/to/.../episode_0000 --hand_side right
#   bash viz_recorded_wuji_qpos_target.sh /path/to/.../episode_0000/data.json --hand_side right
#
# Optional:
#   --save out.mp4   (requires imageio + imageio-ffmpeg)
#   --no_stream      (headless)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ~/miniconda3/bin/activate twist2

python "${SCRIPT_DIR}/deploy_real/viz_recorded_wuji_qpos_target.py" "$@"


