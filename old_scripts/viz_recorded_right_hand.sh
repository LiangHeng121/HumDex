#!/usr/bin/env bash
set -euo pipefail

# Quick viewer for recorded right-hand (hand_tracking_right) in episodes.
# Usage:
#   bash viz_recorded_right_hand.sh /path/to/deploy_real/twist2_demonstration/20260115_0905
#   bash viz_recorded_right_hand.sh /path/to/.../episode_0000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ~/miniconda3/bin/activate twist2

python "${SCRIPT_DIR}/deploy_real/viz_recorded_hand_right.py" "$@"


