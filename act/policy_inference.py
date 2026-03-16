#!/usr/bin/env python3
"""
Policy Inference for HumDex.

Modes:
    1. eval_offline  - Run policy with dataset observations, visualize in sim
    2. eval_online   - Run policy with real-time robot observations (Redis/ZMQ)
    3. init_pose     - Move robot to initial pose from json or demo dataset before eval_online

Usage:
    # Offline evaluation: policy + dataset observations, produces comparison video
    python policy_inference.py eval_offline --ckpt_dir ./ckpt/my_run --dataset ./data/dataset.hdf5

    # Online evaluation: real-time inference on robot
    python policy_inference.py eval_online --ckpt_dir ./ckpt/my_run --temporal_agg

    # Initialize robot pose before online eval
    python policy_inference.py init_pose --init_pose_file ./checkpoints/my_task/init_pose.json 
    python policy_inference.py init_pose --dataset ./data/dataset.hdf5
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Add act/ and scripts/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'act'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.redis_util import RedisIO
from scripts.real_robot_util import VisionReader, KeyboardToggle, ToggleRamp, get_safe_idle_body_35, publish_with_kp_safety
from scripts.policy_util import ACTPolicyWrapper, DatasetReader


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Inference configuration."""
    # Redis
    redis_ip: str = "localhost"
    redis_port: int = 6379
    robot_key: str = "unitree_g1_with_hands"

    # Control
    frequency: float = 30.0  # Hz

    # Policy
    chunk_size: int = 50
    temporal_agg: bool = True
    use_gpu: bool = True
    
    # Hand configuration
    hand_side: str = "left"  # "left", "right", or "both"

    @property
    def state_body_dim(self) -> int:
        return 31
    
    @property
    def hand_dim(self) -> int:
        """Hand dimension: 20 for single hand, 40 for both hands"""
        return 20 if self.hand_side in ["left", "right"] else 40
    
    @property
    def state_dim(self) -> int:
        """Total state dimension: state_body + hand(s)"""
        return self.state_body_dim + self.hand_dim
    
    @property
    def action_dim(self) -> int:
        """Total action dimension: action_body(35) + hand action(s)"""
        return 35 + self.hand_dim  # 55 (single hand) or 75 (both hands)

    # Redis keys
    @property
    def key_state_body(self) -> str:
        return f"state_body_{self.robot_key}"

    @property
    def key_state_hand_left(self) -> str:
        return f"state_wuji_hand_left_{self.robot_key}"
    
    @property
    def key_state_hand_right(self) -> str:
        return f"state_wuji_hand_right_{self.robot_key}"

    @property
    def key_action_body(self) -> str:
        return f"action_body_{self.robot_key}"

    @property
    def key_action_neck(self) -> str:
        return f"action_neck_{self.robot_key}"

    @property
    def key_action_hand_left(self) -> str:
        return f"action_hand_left_{self.robot_key}"

    @property
    def key_action_hand_right(self) -> str:
        return f"action_hand_right_{self.robot_key}"

    @property
    def key_t_action(self) -> str:
        return "t_action"




# =============================================================================
# Offline Eval Mode (Offline evaluation with sim visualization)
# =============================================================================

def eval_offline(
    ckpt_dir: str,
    dataset_path: str,
    episode_id: Optional[int] = None,
    config: Optional[Config] = None,
    policy_config: Optional[dict] = None,
    output_video: Optional[str] = None,
    save_actions: bool = False,
    hand_side: str = "left",
):
    """
    Evaluate policy using observations from dataset, visualize in sim.

    Args:
        ckpt_dir: Path to checkpoint directory
        dataset_path: Path to HDF5 dataset
        episode_id: Episode to evaluate (None = random)
        config: Inference config
        policy_config: Policy architecture config (must match training)
        output_video: Output video path
        save_actions: If True, save predicted actions to .npy file
    """
    config = config or Config()

    # Load dataset
    reader = DatasetReader(dataset_path)
    if episode_id is None:
        episode_id = reader.random_episode_id()

    print(f"\n{'='*60}")
    print(f"Offline Evaluation - Episode {episode_id} (hand: {hand_side})")
    print(f"state_dim: {config.state_dim}")
    print(f"{'='*60}")

    data = reader.load_episode(episode_id, load_observations=True, hand_side=hand_side)
    qpos_all = data['qpos']          # (T, state_dim) - 51 (single hand) or 71 (both)
    images_all = data['images']      # (T, H, W, 3)
    gt_action_body = data['action_body']  # (T, 35) - for comparison
    gt_action_hand = data.get('action_hand', None)  # (T, 20) - for comparison/visualization
    T = data['num_timesteps']

    try:
        text = json.loads(data['text'])
        print(f"Goal: {text.get('goal', 'N/A')}")
    except:
        pass

    print(f"Timesteps: {T}")
    print(f"Observations: qpos={qpos_all.shape}, images={images_all.shape}")

    # Default output path: save under ckpt_dir
    if output_video is None:
        output_video = os.path.join(ckpt_dir, f"eval_ep{episode_id}.mp4")

    # Load policy
    policy = ACTPolicyWrapper(ckpt_dir, config, policy_config)
    policy.reset()

    # Run inference on each timestep
    print(f"\nRunning policy inference...")
    predicted_actions_body = []
    predicted_actions_hand = []
    predicted_actions_full = []

    for t in range(T):
        qpos = qpos_all[t]      # (state_dim,)
        image = images_all[t]   # (H, W, 3)

        action = policy(qpos, image)  # (55,) or (75,) depending on hand_side
        action_body = action[:35]
        action_hand = action[35:]  # (20,) for single hand or (40,) for both hands
        predicted_actions_full.append(action)
        predicted_actions_body.append(action_body)
        predicted_actions_hand.append(action_hand)

        if (t + 1) % 50 == 0 or t == T - 1:
            print(f"  [{t+1}/{T}] pred_z={action_body[2]:.3f}, gt_z={gt_action_body[t, 2]:.3f}")

    predicted_actions_full = np.array(predicted_actions_full)  # (T, 55) or (T, 75)
    predicted_actions_body = np.array(predicted_actions_body)  # (T, 35)
    predicted_actions_hand = np.array(predicted_actions_hand)  # (T, 20) or (T, 40)
    print(f"\nPredicted actions shape: full={predicted_actions_full.shape}, body={predicted_actions_body.shape}, hand={predicted_actions_hand.shape}")

    # Compute error metrics
    mse_body = np.mean((predicted_actions_body - gt_action_body) ** 2)
    mae_body = np.mean(np.abs(predicted_actions_body - gt_action_body))
    print(f"Body  MSE: {mse_body:.6f}, MAE: {mae_body:.6f}")
    if gt_action_hand is not None and gt_action_hand.shape[1] == 20:
        mse_hand = np.mean((predicted_actions_hand - gt_action_hand) ** 2)
        mae_hand = np.mean(np.abs(predicted_actions_hand - gt_action_hand))
        print(f"Hand  MSE: {mse_hand:.6f}, MAE: {mae_hand:.6f}")

    # Save predicted actions
    if save_actions:
        # Backward compatible: body-only arrays
        actions_path = output_video.replace('.mp4', '_actions.npy')
        np.save(actions_path, predicted_actions_body)
        print(f"Saved predicted BODY actions to {actions_path}")

        gt_path = output_video.replace('.mp4', '_gt_actions.npy')
        np.save(gt_path, gt_action_body)
        print(f"Saved GT BODY actions to {gt_path}")

        # New: full + hand arrays
        actions_full_path = output_video.replace('.mp4', '_actions_full.npy')
        np.save(actions_full_path, predicted_actions_full)
        print(f"Saved predicted FULL actions to {actions_full_path}")

        hand_pred_path = output_video.replace('.mp4', f'_actions_hand_{hand_side}.npy')
        np.save(hand_pred_path, predicted_actions_hand)
        print(f"Saved predicted HAND actions to {hand_pred_path}")

        if gt_action_hand is not None:
            hand_gt_path = output_video.replace('.mp4', f'_gt_actions_hand_{hand_side}.npy')
            np.save(hand_gt_path, gt_action_hand)
            print(f"Saved GT HAND actions to {hand_gt_path}")

    # Visualize in sim
    print(f"\nGenerating sim visualization...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'act', 'sim_viz'))
        import cv2
        from visualizers import HumanoidVisualizer, HandVisualizer, get_default_paths, save_video

        paths = get_default_paths()
        viz = HumanoidVisualizer(paths['body_xml'], paths['body_policy'])

        # Helper to add label on every frame
        def add_label(frame, label):
            frame = frame.copy()
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            return frame

        # Visualize predicted/GT BODY actions
        print("Visualizing BODY predicted actions...")
        pred_body_frames = viz.visualize(predicted_actions_body, warmup_steps=100, verbose=False)

        print("Visualizing BODY GT actions...")
        viz._reset()
        gt_body_frames = viz.visualize(gt_action_body, warmup_steps=100, verbose=False)

        # Visualize predicted/GT HAND actions
        if hand_side == "both":
            # Split predicted actions into left (20D) and right (20D)
            predicted_actions_hand_left = predicted_actions_hand[:, :20]   # (T, 20)
            predicted_actions_hand_right = predicted_actions_hand[:, 20:]  # (T, 20)
            
            # Split GT actions into left and right
            if gt_action_hand is not None and gt_action_hand.shape[1] == 40:
                gt_hand_left = gt_action_hand[:, :20]
                gt_hand_right = gt_action_hand[:, 20:]
            else:
                gt_hand_left = np.zeros((T, 20), dtype=np.float32)
                gt_hand_right = np.zeros((T, 20), dtype=np.float32)
            
            # Create left hand visualizer
            hand_xml_key_left = 'left_hand_xml'
            if hand_xml_key_left not in paths:
                raise ValueError(f"Hand XML path for left not found in paths. Available: {list(paths.keys())}")
            hand_viz_left = HandVisualizer(paths[hand_xml_key_left], hand_side='left')
            
            print(f"Visualizing LEFT HAND predicted actions...")
            pred_hand_left_frames = hand_viz_left.visualize(predicted_actions_hand_left, warmup_steps=50, verbose=False)
            
            print(f"Visualizing LEFT HAND GT actions...")
            hand_viz_left._reset()
            gt_hand_left_frames = hand_viz_left.visualize(gt_hand_left, warmup_steps=50, verbose=False)
            
            # Create right hand visualizer
            hand_xml_key_right = 'right_hand_xml'
            if hand_xml_key_right not in paths:
                raise ValueError(f"Hand XML path for right not found in paths. Available: {list(paths.keys())}")
            hand_viz_right = HandVisualizer(paths[hand_xml_key_right], hand_side='right')
            
            print(f"Visualizing RIGHT HAND predicted actions...")
            pred_hand_right_frames = hand_viz_right.visualize(predicted_actions_hand_right, warmup_steps=50, verbose=False)
            
            print(f"Visualizing RIGHT HAND GT actions...")
            hand_viz_right._reset()
            gt_hand_right_frames = hand_viz_right.visualize(gt_hand_right, warmup_steps=50, verbose=False)
            
            # Compose comparison video with both hands:
            #   [Body GT | Body Pred | L-Hand GT | L-Hand Pred | R-Hand GT | R-Hand Pred]
            print("Creating labeled comparison video with both hands...")
            comparison_frames = []
            h = pred_body_frames[0].shape[0]
            w = pred_body_frames[0].shape[1]

            for (bpred, bgt, hlpred, hlgt, hrpred, hrgt) in zip(
                pred_body_frames, gt_body_frames, 
                pred_hand_left_frames, gt_hand_left_frames,
                pred_hand_right_frames, gt_hand_right_frames
            ):
                # Resize hand frames to match body pane size for clean tiling
                hlpred_r = cv2.resize(hlpred, (w, h))
                hlgt_r = cv2.resize(hlgt, (w, h))
                hrpred_r = cv2.resize(hrpred, (w, h))
                hrgt_r = cv2.resize(hrgt, (w, h))

                # Create a 2-row grid: top row has body, bottom row has hands
                top_row = np.concatenate([
                    add_label(bgt, "Body GT"), 
                    add_label(bpred, "Body Pred"),
                    add_label(hlgt_r, "L-Hand GT")
                ], axis=1)
                bottom_row = np.concatenate([
                    add_label(hlpred_r, "L-Hand Pred"),
                    add_label(hrgt_r, "R-Hand GT"),
                    add_label(hrpred_r, "R-Hand Pred")
                ], axis=1)
                grid = np.concatenate([top_row, bottom_row], axis=0)
                comparison_frames.append(grid)
        else:
            # Single hand visualization (original logic)
            hand_xml_key = f'{hand_side}_hand_xml'
            if hand_xml_key not in paths:
                raise ValueError(f"Hand XML path for {hand_side} not found in paths. Available: {list(paths.keys())}")
            hand_viz = HandVisualizer(paths[hand_xml_key], hand_side=hand_side)
            print(f"Visualizing HAND predicted actions ({hand_side})...")
            pred_hand_frames = hand_viz.visualize(predicted_actions_hand, warmup_steps=50, verbose=False)

            print(f"Visualizing HAND GT actions ({hand_side})...")
            hand_viz._reset()
            # If dataset doesn't contain hand GT, this will be zeros (see DatasetReader)
            gt_hand = gt_action_hand if gt_action_hand is not None else np.zeros((T, 20), dtype=np.float32)
            gt_hand_frames = hand_viz.visualize(gt_hand, warmup_steps=50, verbose=False)

            # Compose comparison video as a 2x2 grid with labels:
            #   [Body GT | Body Pred]
            #   [Hand GT | Hand Pred]
            print("Creating labeled 2x2 comparison video...")
            comparison_frames = []
            h = pred_body_frames[0].shape[0]
            w = pred_body_frames[0].shape[1]

            for (bpred, bgt, hpred, hgt) in zip(pred_body_frames, gt_body_frames, pred_hand_frames, gt_hand_frames):
                # Resize hand frames to match body pane size for clean tiling
                hpred_r = cv2.resize(hpred, (w, h))
                hgt_r = cv2.resize(hgt, (w, h))

                top = np.concatenate([add_label(bgt, "Body GT"), add_label(bpred, "Body Pred")], axis=1)
                hand_label = hand_side.upper()[0]  # "L" or "R"
                bottom = np.concatenate([add_label(hgt_r, f"Hand GT ({hand_label})"), add_label(hpred_r, f"Hand Pred ({hand_label})")], axis=1)
                grid = np.concatenate([top, bottom], axis=0)
                comparison_frames.append(grid)

        save_video(comparison_frames, output_video, fps=30)
        print(f"Saved comparison video to {output_video}")

    except Exception as e:
        print(f"Sim visualization failed: {e}")
        print("You can visualize the saved .npy files manually using sim_viz/visualize_body_actions.py")

    print(f"\nEvaluation complete!")


# =============================================================================
# Online Eval Mode (Real-time inference with robot observations)
# =============================================================================

def run_inference(
    ckpt_dir: str,
    config: Optional[Config] = None,
    policy_config: Optional[dict] = None,
    max_timesteps: int = 500,
    vision_ip: str = "192.168.123.164",
    vision_port: int = 5555,
    rgb_stream: bool = False,
    sim_stream: bool = False,
    sim_save_vid: Optional[str] = None,
    sim_hand: bool = False,
    keyboard_toggle_send: bool = False,
    toggle_send_key: str = "k",
    hold_position_key: str = "p",
    record_run: bool = False,
    record_images: bool = False,
    save_rgb_video: bool = False,
    rgb_video_path: Optional[str] = None,
    rgb_video_fps: float = 0.0,
    hand_side: str = "left",
    toggle_ramp_seconds: float = 0.0,
    toggle_ramp_ease: str = "cosine",
):
    """
    Run policy inference with real-time observations.

    Args:
        ckpt_dir: Path to checkpoint directory (contains policy_best.ckpt and dataset_stats.pkl)
        config: Configuration
        policy_config: Policy architecture config (must match training)
        max_timesteps: Maximum number of inference steps
        vision_ip: ZMQ vision server IP
        vision_port: ZMQ vision server port
    """
    config = config or Config()

    print(f"\n{'='*60}")
    print(f"Policy Inference")
    print(f"{'='*60}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Temporal aggregation: {config.temporal_agg}")
    print(f"Frequency: {config.frequency} Hz")
    print(f"Max timesteps: {max_timesteps}")
    print(f"state_dim: {config.state_dim}")

    # Initialize components
    redis_io = RedisIO(config)
    vision = VisionReader(server_ip=vision_ip, port=vision_port)
    policy = ACTPolicyWrapper(ckpt_dir, config, policy_config)

    dt = 1.0 / config.frequency

    # Safe idle action + cached last action (xrobot semantics)
    safe_idle_body_35 = get_safe_idle_body_35(config.robot_key)
    cached_body = safe_idle_body_35.copy()
    cached_neck = np.array([0.0, 0.0], dtype=np.float32)
    last_pub_body = cached_body.copy()
    last_pub_neck = cached_neck.copy()
    ramp = ToggleRamp()

    kb = KeyboardToggle(
        enabled=keyboard_toggle_send,
        toggle_send_key=toggle_send_key,
        hold_position_key=hold_position_key,
    )
    kb.start()
    last_send_enabled, last_hold_enabled = kb.get()

    # -----------------------------------------------------------------------------
    # Optional: record this inference run (qpos + policy actions + published actions)
    # -----------------------------------------------------------------------------
    class _InferRecorder:
        def __init__(self, run_dir: str, max_steps: int, state_dim: int = 51):
            self.run_dir = run_dir
            os.makedirs(self.run_dir, exist_ok=True)

            self.max_steps = int(max_steps)
            self.state_dim = int(state_dim)
            self.i = 0

            # Allocate arrays
            self.ts_ms = np.zeros((self.max_steps,), dtype=np.int64)
            self.send_enabled = np.zeros((self.max_steps,), dtype=np.int8)
            self.hold_enabled = np.zeros((self.max_steps,), dtype=np.int8)

            self.qpos = np.full((self.max_steps, self.state_dim), np.nan, dtype=np.float32)
            self.policy_action_55 = np.full((self.max_steps, 55), np.nan, dtype=np.float32)
            self.pub_body_35 = np.full((self.max_steps, 35), np.nan, dtype=np.float32)
            self.pub_hand_left_20 = np.full((self.max_steps, 20), np.nan, dtype=np.float32)

        def append(
            self,
            ts_ms: int,
            send_enabled: bool,
            hold_enabled: bool,
            qpos: Optional[np.ndarray],
            policy_action_55: Optional[np.ndarray],
            pub_body_35: np.ndarray,
            pub_hand_left_20: Optional[np.ndarray],
        ):
            if self.i >= self.max_steps:
                return
            self.ts_ms[self.i] = int(ts_ms)
            self.send_enabled[self.i] = 1 if send_enabled else 0
            self.hold_enabled[self.i] = 1 if hold_enabled else 0

            if qpos is not None:
                self.qpos[self.i] = np.asarray(qpos, dtype=np.float32).reshape(self.state_dim)
            if policy_action_55 is not None:
                self.policy_action_55[self.i] = np.asarray(policy_action_55, dtype=np.float32).reshape(55)

            self.pub_body_35[self.i] = np.asarray(pub_body_35, dtype=np.float32).reshape(35)
            if pub_hand_left_20 is not None:
                self.pub_hand_left_20[self.i] = np.asarray(pub_hand_left_20, dtype=np.float32).reshape(20)

            self.i += 1

        def maybe_save_image(self, img_rgb: np.ndarray):
            if not record_images:
                return
            try:
                import cv2
                img_bgr = img_rgb[:, :, ::-1]
                cv2.imwrite(os.path.join(self.run_dir, f"rgb_{self.i:06d}.jpg"), img_bgr)
            except Exception:
                pass

        def close(self, meta: dict):
            # Truncate to actual length and save
            n = self.i
            np.save(os.path.join(self.run_dir, "ts_ms.npy"), self.ts_ms[:n])
            np.save(os.path.join(self.run_dir, "send_enabled.npy"), self.send_enabled[:n])
            np.save(os.path.join(self.run_dir, "hold_enabled.npy"), self.hold_enabled[:n])
            # Save qpos with state_dim suffix for clarity
            np.save(os.path.join(self.run_dir, f"qpos_{self.state_dim}.npy"), self.qpos[:n])
            np.save(os.path.join(self.run_dir, "policy_action_55.npy"), self.policy_action_55[:n])
            np.save(os.path.join(self.run_dir, "pub_body_35.npy"), self.pub_body_35[:n])
            np.save(os.path.join(self.run_dir, "pub_hand_left_20.npy"), self.pub_hand_left_20[:n])
            with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

    recorder = None
    run_dir = None
    if record_run:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(ckpt_dir, "eval", f"this_run_{ts}")
        print(f"[Record] Enabled. Saving to: {run_dir}")
        recorder = _InferRecorder(run_dir, max_steps=max_timesteps, state_dim=config.state_dim)

    # -----------------------------------------------------------------------------
    # Optional: save RealSense RGB stream to mp4 under ckpt_dir
    # -----------------------------------------------------------------------------
    rgb_writer = None
    rgb_save_path = None
    if save_rgb_video:
        try:
            import cv2
            ts = time.strftime("%Y%m%d_%H%M%S")
            # default: save under ckpt_dir/eval/
            out_dir = os.path.join(ckpt_dir, "eval")
            os.makedirs(out_dir, exist_ok=True)
            rgb_save_path = rgb_video_path
            if not rgb_save_path:
                rgb_save_path = os.path.join(out_dir, f"infer_rgb_{ts}.mp4")
            fps = float(rgb_video_fps) if float(rgb_video_fps) > 1e-6 else float(config.frequency)
            # VisionReader default is 480x640 RGB
            h, w = int(vision.img_shape[0]), int(vision.img_shape[1])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            rgb_writer = cv2.VideoWriter(rgb_save_path, fourcc, fps, (w, h))
            if not rgb_writer.isOpened():
                raise RuntimeError(f"VideoWriter failed to open: {rgb_save_path}")
            print(f"[RGB] Recording enabled: {rgb_save_path} (fps={fps:.2f}, size={w}x{h})")
        except Exception as e:
            print(f"[RGB] Failed to start video recording: {e}")
            rgb_writer = None
            rgb_save_path = None

    print(f"\nWaiting for state data... (expected state_dim={config.state_dim})")
    while True:
        qpos = redis_io.read_state()
        if qpos is not None:
            print(f"Got initial state: shape={qpos.shape}")
            break
        time.sleep(0.1)

    # Optional windows
    # If running headless (no DISPLAY/WAYLAND), disable pop-up windows to avoid blocking.
    if (rgb_stream or sim_stream) and (os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None):
        print("[GUI] No DISPLAY/WAYLAND_DISPLAY detected; disabling --rgb_stream/--sim_stream (use --sim_save_vid instead).")
        rgb_stream = False
        sim_stream = False

    if rgb_stream:
        try:
            print("[RGB] Opening Robot RGB window...")
            import cv2
            cv2.namedWindow("Robot RGB", cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"[RGB] Failed to open RGB window: {e}")
            rgb_stream = False

    sim_viz = None
    sim_writer = None
    hand_viz = None
    hand_viz_right = None
    cached_hand_20 = np.zeros(20, dtype=np.float32)
    cached_hand_20_right = np.zeros(20, dtype=np.float32)
    use_both_hands = (hand_side == "both")
    
    if sim_stream or sim_save_vid:
        try:
            print("[Sim] Initializing sim preview...")
            import cv2
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'act', 'sim_viz'))
            from visualizers import HumanoidVisualizer, HandVisualizer, get_default_paths
            paths = get_default_paths()
            sim_viz = HumanoidVisualizer(paths['body_xml'], paths['body_policy'])
            if sim_hand:
                if use_both_hands:
                    # Initialize both hand visualizers
                    hand_xml_key_left = 'left_hand_xml'
                    hand_xml_key_right = 'right_hand_xml'
                    if hand_xml_key_left not in paths or hand_xml_key_right not in paths:
                        raise ValueError(f"Hand XML paths not found in paths. Available: {list(paths.keys())}")
                    hand_viz = HandVisualizer(paths[hand_xml_key_left], hand_side='left')
                    hand_viz_right = HandVisualizer(paths[hand_xml_key_right], hand_side='right')
                else:
                    # Single hand
                    hand_xml_key = f'{hand_side}_hand_xml'
                    if hand_xml_key not in paths:
                        raise ValueError(f"Hand XML path for {hand_side} not found in paths. Available: {list(paths.keys())}")
                    hand_viz = HandVisualizer(paths[hand_xml_key], hand_side=hand_side)
            if sim_stream:
                print("[Sim] Opening Sim (Body) window...")
                cv2.namedWindow("Sim (Body)", cv2.WINDOW_NORMAL)
            if sim_save_vid:
                if sim_save_vid == "__AUTO__":
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    sim_save_vid = os.path.join(ckpt_dir, f"infer_sim_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                if use_both_hands and sim_hand:
                    out_w, out_h = (1920, 480)  # body + left hand + right hand
                else:
                    out_w, out_h = (1280, 480) if sim_hand else (640, 480)
                sim_writer = cv2.VideoWriter(sim_save_vid, fourcc, 30, (out_w, out_h))
                if not sim_writer.isOpened():
                    raise RuntimeError(f"VideoWriter failed to open: {sim_save_vid}")
                print(f"[Sim] Recording enabled: {sim_save_vid}")
        except Exception as e:
            print(f"[Sim] Failed to start sim visualization: {e}")
            sim_viz = None
            sim_writer = None
            hand_viz = None
            hand_viz_right = None

    print(f"\nRunning inference... (Ctrl+C to stop)\n")

    try:
        for t in range(max_timesteps):
            t0 = time.time()

            # Always update RGB window from live vision stream
            live_image = vision.get_image()
            if rgb_stream:
                import cv2
                cv2.imshow("Robot RGB", live_image[:, :, ::-1])
                cv2.waitKey(1)
            if rgb_writer is not None:
                try:
                    import cv2
                    # VideoWriter expects BGR
                    rgb_writer.write(cv2.cvtColor(live_image, cv2.COLOR_RGB2BGR))
                except Exception:
                    pass

            send_enabled, hold_enabled = kb.get()

            # Detect k/p transitions and ramp like init_pose (freeze policy/time during ramp)
            if (send_enabled != last_send_enabled) or (hold_enabled != last_hold_enabled):
                # Determine target for NEW mode
                if not send_enabled:
                    target_body = safe_idle_body_35
                    target_neck = cached_neck
                    target_mode = "default"
                    # Also reset any policy temporal state when we go to idle (safer)
                    try:
                        policy.reset()
                    except Exception:
                        pass
                elif hold_enabled:
                    target_body = cached_body
                    target_neck = cached_neck
                    target_mode = "hold"
                else:
                    qpos_now = redis_io.read_state()
                    if qpos_now is None:
                        target_body = cached_body
                        target_neck = cached_neck
                    else:
                        _a = policy(qpos_now, live_image)
                        target_body = _a[:35]
                        target_neck = cached_neck
                    target_mode = "follow"

                ramp.start(
                    from_body=last_pub_body,
                    from_neck=last_pub_neck,
                    to_body=target_body,
                    to_neck=target_neck,
                    target_mode=target_mode,
                    seconds=float(toggle_ramp_seconds),
                    ease=str(toggle_ramp_ease),
                )
                last_send_enabled, last_hold_enabled = send_enabled, hold_enabled

            # If ramping, publish interpolated action and skip policy/time advance
            if ramp.active:
                pub_body, pub_neck, _done = ramp.value()
                try:
                    redis_io.set_wuji_hand_mode(ramp.target_mode)
                except Exception:
                    pass
                redis_io.publish_action(pub_body, pub_neck)
                last_pub_body = np.asarray(pub_body, dtype=np.float32).reshape(35).copy()
                last_pub_neck = np.asarray(pub_neck, dtype=np.float32).reshape(2).copy()

                # keep UI/recording alive but don't run policy
                if recorder is not None:
                    now = int(time.time() * 1000)
                    recorder.append(
                        ts_ms=now,
                        send_enabled=send_enabled,
                        hold_enabled=hold_enabled,
                        qpos=None,
                        policy_action_55=None,
                        pub_body_35=pub_body,
                        pub_hand_left_20=cached_hand_20,
                    )
                    recorder.maybe_save_image(live_image)

                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue

            # Only compute desired action if we are not in k/p override states
            desired_body = cached_body
            desired_neck = cached_neck
            desired_hand_20 = None
            policy_action_full_55 = None
            obs_qpos = None
            if send_enabled and (not hold_enabled):
                qpos = redis_io.read_state()
                if qpos is None:
                    print(f"  [{t}] Warning: no state data")
                    time.sleep(dt)
                    continue
                action = policy(qpos, live_image)  # (55,)
                desired_body = action[:35]
                desired_hand_20 = action[35:]
                policy_action_full_55 = action
                obs_qpos = qpos

            pub_body, pub_neck, cached_body, cached_neck, advance = publish_with_kp_safety(
                redis_io=redis_io,
                kb=kb,
                safe_idle_body_35=safe_idle_body_35,
                cached_body=cached_body,
                cached_neck=cached_neck,
                desired_body=desired_body,
                desired_neck=desired_neck,
            )

            # Publish Wuji hand 20D target when we actually advanced/ran policy
            if advance and desired_hand_20 is not None:
                if use_both_hands:
                    # Split into left (20D) and right (20D)
                    hand_left = desired_hand_20[:20]
                    hand_right = desired_hand_20[20:]
                    redis_io.publish_wuji_qpos_target(hand_left, hand_side='left')
                    redis_io.publish_wuji_qpos_target(hand_right, hand_side='right')
                    cached_hand_20 = np.asarray(hand_left, dtype=np.float32).reshape(-1)
                    cached_hand_20_right = np.asarray(hand_right, dtype=np.float32).reshape(-1)
                else:
                    redis_io.publish_wuji_qpos_target(desired_hand_20, hand_side=hand_side)
                    cached_hand_20 = np.asarray(desired_hand_20, dtype=np.float32).reshape(-1)

            # Record (always record what we published; policy_action may be NaN if in k/p override)
            if recorder is not None:
                now = int(time.time() * 1000)
                recorder.append(
                    ts_ms=now,
                    send_enabled=send_enabled,
                    hold_enabled=hold_enabled,
                    qpos=obs_qpos,
                    policy_action_55=policy_action_full_55,
                    pub_body_35=pub_body,
                    pub_hand_left_20=cached_hand_20,
                )
                recorder.maybe_save_image(live_image)

            # Sim preview
            if sim_viz is not None:
                import cv2
                
                # Helper to add label
                def add_label(frame, label):
                    frame = frame.copy()
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    return frame
                
                body_frame = sim_viz.step(np.asarray(pub_body, dtype=np.float32))
                out_frame = add_label(body_frame, "Body")
                
                if hand_viz is not None:
                    hand_frame = hand_viz.step(cached_hand_20)
                    hand_frame = add_label(hand_frame, "L-Hand" if use_both_hands else f"{hand_side[0].upper()}-Hand")
                    
                    if use_both_hands and hand_viz_right is not None:
                        hand_frame_right = hand_viz_right.step(cached_hand_20_right)
                        hand_frame_right = add_label(hand_frame_right, "R-Hand")
                        out_frame = np.concatenate([out_frame, hand_frame, hand_frame_right], axis=1)
                    else:
                        out_frame = np.concatenate([out_frame, hand_frame], axis=1)
                        
                if sim_stream:
                    cv2.imshow("Sim (Body)", out_frame[:, :, ::-1])
                    cv2.waitKey(1)
                if sim_writer is not None:
                    sim_writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

            # Progress
            if (t + 1) % 30 == 0:
                print(f"  [{t+1}/{max_timesteps}] z={pub_body[2]:.3f}")

            # Rate limiting
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

            last_pub_body = np.asarray(pub_body, dtype=np.float32).reshape(35).copy()
            last_pub_neck = np.asarray(pub_neck, dtype=np.float32).reshape(2).copy()

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        kb.stop()
        vision.close()
        try:
            import cv2
            if rgb_writer is not None:
                rgb_writer.release()
                if rgb_save_path:
                    print(f"[RGB] Saved video to {rgb_save_path}")
            if sim_writer is not None:
                sim_writer.release()
            cv2.destroyAllWindows()
        except Exception:
            pass

        if recorder is not None and run_dir is not None:
            meta = {
                "ckpt_dir": ckpt_dir,
                "frequency": config.frequency,
                "chunk_size": config.chunk_size,
                "temporal_agg": config.temporal_agg,
                "vision_ip": vision_ip,
                "vision_port": vision_port,
                "max_timesteps": max_timesteps,
                "saved_images": bool(record_images),
                "saved_rgb_video": str(rgb_save_path) if rgb_save_path else "",
                "state_dim": config.state_dim,
            }
            recorder.close(meta)
            print(f"[Record] Saved {recorder.i} steps to: {run_dir}")

    print(f"Inference complete!")


def publish_initial_action(
    config: Optional[Config] = None,
    dataset_path: Optional[str] = None,
    init_pose_file: Optional[str] = None,
    episode_id: Optional[int] = None,
    timestep: int = 0,
    keyboard_toggle_send: bool = True,
    toggle_send_key: str = "k",
    hold_position_key: str = "p",
    hand_side: str = "left",
    ramp_seconds: float = 0.0,
    ramp_ease: str = "cosine",
    ramp_from: str = "redis_action",
):
    """Continuously publish an initial action to move robot to initial pose.

    Loads the target pose from either an init_pose JSON file or an HDF5 dataset.
    Intended usage: run this first, watch robot reach initial pose, Ctrl-C to stop,
    then run `eval_online`.
    """
    config = config or Config()

    if init_pose_file is not None:
        with open(init_pose_file, 'r') as f:
            pose = json.load(f)
        body = np.asarray(pose["action_body"], dtype=np.float32)
        hand_side = pose.get("hand_side", hand_side)
        hand_parts = []
        if hand_side in ("left", "both") and "action_hand_left" in pose:
            hand_parts.append(np.asarray(pose["action_hand_left"], dtype=np.float32))
        if hand_side in ("right", "both") and "action_hand_right" in pose:
            hand_parts.append(np.asarray(pose["action_hand_right"], dtype=np.float32))
        hand_20 = np.concatenate(hand_parts) if hand_parts else None

        print(f"\n{'='*60}")
        print("Publish initial action (hold) from JSON")
        print(f"{'='*60}")
        print(f"File: {init_pose_file}")
        print(f"hand_side: {hand_side}")
    elif dataset_path is not None:
        reader = DatasetReader(dataset_path)
        if episode_id is None:
            episode_id = reader.random_episode_id()
        data = reader.load_episode(episode_id, load_observations=False, hand_side=hand_side)
        T = int(data["num_timesteps"])
        timestep = int(np.clip(timestep, 0, T - 1))
        body = np.asarray(data["action_body"][timestep], dtype=np.float32)
        hand_20 = None
        if "action_hand" in data:
            try:
                hand_20 = np.asarray(data["action_hand"][timestep], dtype=np.float32)
            except Exception:
                hand_20 = None

        print(f"\n{'='*60}")
        print("Publish initial action (hold) from dataset")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_path}")
        print(f"Episode: {episode_id} (T={T})")
        print(f"Timestep: {timestep}")
    else:
        raise ValueError("Either --init_pose_file or --dataset is required for init_pose mode")

    print(f"Publishing to Redis at {config.redis_ip}:{config.redis_port} robot_key={config.robot_key}")
    if float(ramp_seconds) > 0.0:
        print(f"Ramp: enabled ({float(ramp_seconds):.2f}s, ease={ramp_ease}, from={ramp_from})")
    print("Ctrl-C to stop.")

    redis_io = RedisIO(config)
    dt = 1.0 / config.frequency

    safe_idle_body_35 = get_safe_idle_body_35(config.robot_key)
    cached_body = body.copy()
    cached_neck = np.zeros(2, dtype=np.float32)

    # ---------------------------
    # Optional: smooth ramp-in
    # ---------------------------
    def _ease(alpha: float) -> float:
        a = float(np.clip(alpha, 0.0, 1.0))
        if ramp_ease == "linear":
            return a
        # cosine ease-in-out
        return 0.5 - 0.5 * float(np.cos(np.pi * a))

    def _read_redis_json_array(key: str, expected_len: Optional[int] = None) -> Optional[np.ndarray]:
        try:
            raw = redis_io.client.get(key)
            arr = redis_io._safe_json_load(raw)
            if arr is None:
                return None
            out = np.asarray(arr, dtype=np.float32).reshape(-1)
            if expected_len is not None and out.shape[0] != int(expected_len):
                return None
            return out
        except Exception:
            return None

    start_body = None
    start_hand = None
    if float(ramp_seconds) > 0.0:
        if ramp_from == "redis_action":
            start_body = _read_redis_json_array(config.key_action_body, expected_len=35)
            if hand_20 is not None:
                start_hand = _read_redis_json_array(
                    f"action_wuji_qpos_target_{hand_side}_{config.robot_key}",
                    expected_len=20,
                )

        if start_body is None:
            start_body = safe_idle_body_35.copy()
        if hand_20 is not None and start_hand is None:
            start_hand = np.zeros(20, dtype=np.float32)

    ramp_t0 = time.time()
    kb = KeyboardToggle(
        enabled=keyboard_toggle_send,
        toggle_send_key=toggle_send_key,
        hold_position_key=hold_position_key,
    )
    kb.start()

    try:
        while True:
            t0 = time.time()
            desired_body = body
            desired_hand = hand_20

            if float(ramp_seconds) > 0.0:
                alpha = (time.time() - ramp_t0) / max(1e-6, float(ramp_seconds))
                w = _ease(alpha)
                desired_body = start_body + w * (body - start_body)
                if hand_20 is not None and start_hand is not None:
                    desired_hand = start_hand + w * (hand_20 - start_hand)

            _pub_body, _pub_neck, cached_body, cached_neck, _advance = publish_with_kp_safety(
                redis_io=redis_io,
                kb=kb,
                safe_idle_body_35=safe_idle_body_35,
                cached_body=cached_body,
                cached_neck=cached_neck,
                desired_body=desired_body,
            )
            if desired_hand is not None:
                redis_io.publish_wuji_qpos_target(desired_hand, hand_side=hand_side)
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        kb.stop()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TWIST2 Policy Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # eval_offline - run policy with dataset observations, visualize in sim
    eval_parser = subparsers.add_parser('eval_offline', help='Evaluate policy with dataset observations (sim visualization)')
    eval_parser.add_argument('--ckpt_dir', required=True, help='Checkpoint directory')
    eval_parser.add_argument('--dataset', required=True, help='Path to HDF5 dataset')
    eval_parser.add_argument('--episode', type=int, default=None, help='Episode ID (default: random)')
    eval_parser.add_argument('--output', default=None, help='Output video path (default: ckpt_dir/eval_ep{episode}.mp4)')
    eval_parser.add_argument('--temporal_agg', action='store_true', help='Use temporal aggregation')
    eval_parser.add_argument('--chunk_size', type=int, default=50, help='Action chunk size (must match training)')
    eval_parser.add_argument('--hidden_dim', type=int, default=512, help='Policy hidden dim (must match training)')
    eval_parser.add_argument('--dim_feedforward', type=int, default=3200,
                             help='Transformer FFN dim (must match training, e.g. 3200)')
    eval_parser.add_argument('--save_actions', action='store_true',
                             help='If set, save predicted/GT actions to .npy alongside the output video')
    eval_parser.add_argument('--hand_side', type=str, default='left', choices=['left', 'right', 'both'],
                             help='Which hand to use: left, right, or both (default: left)')
    eval_parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')

    # eval_online - run policy with real-time robot observations
    online_parser = subparsers.add_parser('eval_online', help='Run policy with real-time robot observations')
    online_parser.add_argument('--ckpt_dir', required=True, help='Checkpoint directory')
    online_parser.add_argument('--redis_ip', default='localhost', help='Redis IP')
    online_parser.add_argument('--frequency', type=float, default=30.0, help='Control frequency (Hz)')
    online_parser.add_argument('--max_timesteps', type=int, default=500, help='Max inference steps')
    online_parser.add_argument('--temporal_agg', action='store_true', help='Use temporal aggregation')
    online_parser.add_argument('--chunk_size', type=int, default=50, help='Action chunk size (must match training)')
    online_parser.add_argument('--hidden_dim', type=int, default=512, help='Policy hidden dim (must match training)')
    online_parser.add_argument('--dim_feedforward', type=int, default=3200,
                              help='Transformer FFN dim (must match training, e.g. 3200)')
    online_parser.add_argument('--vision_ip', default='192.168.123.164', help='Vision server IP')
    online_parser.add_argument('--vision_port', type=int, default=5555, help='Vision server port')
    online_parser.add_argument('--rgb_stream', action='store_true', help='Open a window to stream robot RGB')
    online_parser.add_argument('--sim_stream', action='store_true', help='Open a window to stream sim preview (body)')
    online_parser.add_argument('--sim_save_vid', nargs='?', const="__AUTO__", default=None,
                              help='Save sim preview mp4 (optionally provide path)')
    online_parser.add_argument('--sim_hand', action='store_true', help='Include hand visualization in sim preview/video')
    online_parser.add_argument('--hand_side', type=str, default='left', choices=['left', 'right', 'both'],
                              help='Which hand to use: left, right, or both (default: left)')
    online_parser.set_defaults(keyboard_toggle_send=True)
    online_parser.add_argument('--no_keyboard_toggle_send', dest='keyboard_toggle_send', action='store_false',
                              help="Disable terminal keyboard safety toggles (NOT recommended on real robot)")
    online_parser.add_argument('--toggle_send_key', type=str, default='k', help="Key to toggle send_enabled (default 'k')")
    online_parser.add_argument('--hold_position_key', type=str, default='p', help="Key to toggle hold_position (default 'p')")
    online_parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    online_parser.add_argument('--record_run', action='store_true',
                              help='Record qpos + policy actions + published actions under ckpt_dir/eval/')
    online_parser.add_argument('--record_images', action='store_true',
                              help='If set with --record_run, also save live RGB frames as JPGs')
    online_parser.add_argument('--save_rgb_video', action='store_true',
                              help='Save live RGB stream to mp4 under ckpt_dir/eval/')
    online_parser.add_argument('--rgb_video_path', default=None,
                              help='Optional output path for RGB mp4')
    online_parser.add_argument('--rgb_video_fps', type=float, default=0.0,
                              help='FPS for saved RGB video (default: use --frequency)')
    online_parser.add_argument('--toggle_ramp_seconds', type=float, default=0.0,
                              help='Smooth interpolation duration on k/p toggles (seconds, 0 to disable)')
    online_parser.add_argument('--toggle_ramp_ease', type=str, default='cosine', choices=['linear', 'cosine'],
                              help='Toggle ramp easing curve (default: cosine)')

    # init_pose - publish a fixed initial action until Ctrl-C
    init_parser = subparsers.add_parser('init_pose', help='Move robot to initial pose (hold until Ctrl-C)')
    init_pose_src = init_parser.add_mutually_exclusive_group(required=True)
    init_pose_src.add_argument('--init_pose_file', help='Path to init_pose.json')
    init_pose_src.add_argument('--dataset', help='Path to HDF5 dataset (alternative to --init_pose_file)')
    init_parser.add_argument('--episode', type=int, default=None, help='Episode ID (default: random, only with --dataset)')
    init_parser.add_argument('--timestep', type=int, default=0, help='Timestep index (default: 0, only with --dataset)')
    init_parser.add_argument('--redis_ip', default='localhost', help='Redis IP')
    init_parser.add_argument('--frequency', type=float, default=30.0, help='Publish frequency (Hz)')
    init_parser.add_argument('--robot_key', default='unitree_g1_with_hands', help='Robot key suffix for Redis keys')
    init_parser.add_argument('--hand_side', type=str, default='left', choices=['left', 'right', 'both'],
                             help='Which hand to use: left, right, or both (default: left)')
    init_parser.add_argument('--ramp_seconds', type=float, default=0.0,
                             help='Smoothly interpolate from current action to target over this duration (seconds, 0 disables)')
    init_parser.add_argument('--ramp_ease', type=str, default='cosine', choices=['linear', 'cosine'],
                             help='Ramp easing curve (default: cosine)')
    init_parser.add_argument('--ramp_from', type=str, default='redis_action', choices=['redis_action', 'safe_idle'],
                             help='Ramp start source: redis_action (current published action) or safe_idle')
    init_parser.set_defaults(keyboard_toggle_send=True)
    init_parser.add_argument('--no_keyboard_toggle_send', dest='keyboard_toggle_send', action='store_false',
                             help="Disable terminal keyboard safety toggles (NOT recommended on real robot)")
    init_parser.add_argument('--toggle_send_key', type=str, default='k', help="Key to toggle send_enabled (default 'k')")
    init_parser.add_argument('--hold_position_key', type=str, default='p', help="Key to toggle hold_position (default 'p')")

    args = parser.parse_args()

    if args.mode == 'eval_offline':
        config = Config(
            temporal_agg=args.temporal_agg,
            chunk_size=args.chunk_size,
            use_gpu=not args.cpu,
            hand_side=args.hand_side,
        )
        policy_config = {
            'lr': 1e-5,
            'lr_backbone': 1e-5,
            'num_queries': args.chunk_size,
            'kl_weight': 10,
            'hidden_dim': args.hidden_dim,
            'dim_feedforward': args.dim_feedforward,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': ['head'],
        }
        eval_offline(
            ckpt_dir=args.ckpt_dir,
            dataset_path=args.dataset,
            episode_id=args.episode,
            config=config,
            policy_config=policy_config,
            output_video=args.output,
            save_actions=args.save_actions,
            hand_side=args.hand_side,
        )

    elif args.mode == 'eval_online':
        config = Config(
            redis_ip=args.redis_ip,
            frequency=args.frequency,
            temporal_agg=args.temporal_agg,
            chunk_size=args.chunk_size,
            use_gpu=not args.cpu,
            hand_side=args.hand_side,
        )
        policy_config = {
            'lr': 1e-5,
            'lr_backbone': 1e-5,
            'num_queries': args.chunk_size,
            'kl_weight': 10,
            'hidden_dim': args.hidden_dim,
            'dim_feedforward': args.dim_feedforward,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': ['head'],
        }
        run_inference(
            ckpt_dir=args.ckpt_dir,
            config=config,
            policy_config=policy_config,
            max_timesteps=args.max_timesteps,
            vision_ip=args.vision_ip,
            vision_port=args.vision_port,
            rgb_stream=args.rgb_stream,
            sim_stream=args.sim_stream,
            sim_save_vid=args.sim_save_vid,
            sim_hand=args.sim_hand,
            keyboard_toggle_send=args.keyboard_toggle_send,
            toggle_send_key=args.toggle_send_key,
            hold_position_key=args.hold_position_key,
            record_run=args.record_run,
            record_images=args.record_images,
            save_rgb_video=bool(args.save_rgb_video),
            rgb_video_path=args.rgb_video_path,
            rgb_video_fps=float(args.rgb_video_fps),
            hand_side=args.hand_side,
            toggle_ramp_seconds=float(args.toggle_ramp_seconds),
            toggle_ramp_ease=str(args.toggle_ramp_ease),
        )

    elif args.mode == 'init_pose':
        config = Config(
            redis_ip=args.redis_ip,
            frequency=args.frequency,
            robot_key=args.robot_key,
        )
        publish_initial_action(
            config=config,
            dataset_path=args.dataset,
            init_pose_file=getattr(args, 'init_pose_file', None),
            episode_id=args.episode,
            timestep=args.timestep,
            keyboard_toggle_send=args.keyboard_toggle_send,
            toggle_send_key=args.toggle_send_key,
            hold_position_key=args.hold_position_key,
            hand_side=args.hand_side,
            ramp_seconds=args.ramp_seconds,
            ramp_ease=args.ramp_ease,
            ramp_from=args.ramp_from,
        )


if __name__ == "__main__":
    main()
