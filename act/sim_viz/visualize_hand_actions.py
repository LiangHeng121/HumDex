#!/usr/bin/env python3
"""Visualize predicted 20D hand actions in MuJoCo sim."""

import argparse
import numpy as np
from pathlib import Path
from visualizers import HandVisualizer, get_default_paths


def main():
    defaults = get_default_paths()

    parser = argparse.ArgumentParser(description='Visualize 20D hand actions')
    parser.add_argument('--actions', required=True, help='Path to (T, 20) .npy file')
    parser.add_argument('--hand_side', required=True, choices=['left', 'right'])
    parser.add_argument('--xml', default=None, help='Hand MuJoCo XML (auto-detected if omitted)')
    parser.add_argument('--output', default=None, help='Output video path')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    print(f"Loading hand actions: {args.actions}")
    actions = np.load(args.actions)
    print(f"  Shape: {actions.shape}")

    if actions.ndim == 1:
        if len(actions) != 20:
            raise ValueError(f"Expected 20D action, got {len(actions)}D")
        actions = actions.reshape(1, -1)
    elif actions.ndim == 3:
        actions = actions.reshape(len(actions), -1)
    elif actions.ndim == 2 and actions.shape[1] != 20:
        raise ValueError(f"Expected (T, 20) actions, got {actions.shape}")

    hand_xml = args.xml or defaults[f"{args.hand_side}_hand_xml"]
    if not Path(hand_xml).exists():
        print(f"Error: XML file not found: {hand_xml}")
        return 1

    viz = HandVisualizer(
        xml_path=hand_xml, hand_side=args.hand_side,
        width=args.width, height=args.height
    )
    viz.visualize(actions, output_video=args.output, fps=args.fps)


if __name__ == "__main__":
    main()
