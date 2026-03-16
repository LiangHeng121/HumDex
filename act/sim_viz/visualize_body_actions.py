#!/usr/bin/env python3
"""Visualize predicted 35D body actions in MuJoCo sim."""

import argparse
import numpy as np
from visualizers import HumanoidVisualizer, get_default_paths


def main():
    defaults = get_default_paths()

    parser = argparse.ArgumentParser(description='Visualize 35D body actions')
    parser.add_argument('--actions', required=True, help='Path to (T, 35) .npy file')
    parser.add_argument('--policy', default=defaults["body_policy"], help='ONNX RL policy path')
    parser.add_argument('--xml', default=defaults["body_xml"], help='MuJoCo XML model path')
    parser.add_argument('--output', default=None, help='Output video path')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    print(f"Loading actions: {args.actions}")
    actions = np.load(args.actions)
    print(f"  Shape: {actions.shape}")

    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[1] != 35:
        raise ValueError(f"Expected (T, 35) actions, got {actions.shape}")

    viz = HumanoidVisualizer(
        xml_path=args.xml, policy_path=args.policy,
        device=args.device, width=args.width, height=args.height
    )
    viz.visualize(actions, output_video=args.output, fps=args.fps,
                  warmup_steps=args.warmup_steps)


if __name__ == "__main__":
    main()
