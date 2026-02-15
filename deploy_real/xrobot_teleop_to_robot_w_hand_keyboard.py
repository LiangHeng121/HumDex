"""
conda activate gmr
sudo ufw disable
python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1

State Machine Controls:
- Right controller key_one: Cycle through idle -> teleop -> pause -> teleop...
- Left controller key_one: Exit program from any state
- Left controller axis_click: Emergency stop - kills sim2real.sh process
- Left controller axis: Control root xy velocity and yaw velocity
- Right controller axis: Fine-tune root xy velocity and yaw velocity
- Auto-transition: idle -> teleop when motion data is available

States:
- idle: Waiting for input or data
- teleop: Processing motion retargeting with velocity control
- pause: Data received but not processing
- exit: Program will terminate

Whole-Body Teleop Features:
- Sends whole-body mode information to Redis
- 35-dimensional mimic observations
- Uses retargeted motion directly from the teleoperation stream
"""
import argparse
import json
import pathlib
import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import draw_frame
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT
from general_motion_retargeting import human_head_to_robot_neck
from rich import print
from tqdm import tqdm
import cv2
import redis
from rich import print
from general_motion_retargeting import XRobotStreamer

from data_utils.params import DEFAULT_MIMIC_OBS, DEFAULT_HAND_POSE
from data_utils.rot_utils import euler_from_quaternion_np, quat_diff_np, quat_rotate_inverse_np
from data_utils.fps_monitor import FPSMonitor
from data_utils.evdev_hotkeys import EvdevHotkeyConfig, EvdevHotkeys

# ---------------------------------------------------------------------
# Safe idle (k: send_enabled=False) presets
# ---------------------------------------------------------------------
# When keyboard toggle is OFF (k), we publish a "safe idle" 35D mimic_obs.
# Select via CLI: --safe_idle_pose_id 0/1/2...
#
# NOTE: Must be length 35.
SAFE_IDLE_BODY_35_PRESETS = {
    # 0: original repo default (keeps backward compatibility)
    0: [
        -2.962986573041272e-06,
        6.836185035045111e-06,
        0.7900107971067252,
        0.026266981563484476,
        -0.07011304233181229,
        -0.00038564063739400495,
        0.21007653006396093,
        0.1255744557454361,
        0.5210019779740723,
        -0.087267,
        0.023696508296266388,
        -0.12259741578159437,
        0.18640974335249333,
        -0.1213838414703421,
        0.11017991599235927,
        -0.087267,
        -0.06074348170695354,
        0.10802879748679631,
        -0.14690420989255235,
        -0.06195140749854128,
        0.03492134295105836,
        -0.012934516116481467,
        0.012973065503571952,
        -0.09877424821663634,
        1.5735338678105346,
        -0.08846852951921763,
        -0.008568943127155513,
        -0.07037145190015832,
        -0.45191594425028536,
        -0.7548272891300677,
        0.07631181877180071,
        0.623873998918081,
        0.32440260037889024,
        -0.17081521970550126,
        0.2697219398563502,
    ],
    # 1: captured from Redis (action source) on your machine (2026-01-17)
    1: [
        -0.002425597382764927,
        0.0004014222794810171,
        0.789948578249186,
        -0.05286645234860116,
        -0.11395774381848182,
        -0.0020091780029543797,
        0.33550286925644013,
        0.07678254800339449,
        -0.11831599235723278,
        -0.087267,
        -0.1536621162766681,
        -0.039016535005063684,
        0.28263936593666483,
        -0.01999487086573224,
        -0.3918089438082317,
        -0.08726699999999998,
        -0.06775504509688593,
        0.0727761475591654,
        -0.09677870600760852,
        -0.0027568505266116657,
        0.07348304585982098,
        -0.10334908779279858,
        0.3160389030446376,
        0.07844298473038674,
        1.3008225711954524,
        0.6130673022421114,
        -0.2198179601159421,
        0.3438907117467236,
        -0.23448010297908417,
        -0.5483439694277361,
        -0.3146753829836872,
        0.910606700768848,
        -0.22716316478096404,
        -0.10501071874258898,
        -0.2864687400817216,
    ],
    2: [
        0.0, 0.0, 0.79, 0.004581602464116093, 0.054385222258041876, -0.01047197449952364,
        -0.1705406904220581, -0.011608824133872986, -0.08608310669660568, 0.2819371521472931,
        -0.13509835302829742, 0.028368590399622917, -0.15945219993591309, -0.011438383720815182,
        0.09397093206644058, 0.2500985264778137, -0.12299267947673798, 0.033810943365097046,
        0.01984678953886032, 0.04372693970799446, 0.04439987987279892, -0.052922338247299194,
        0.3638530671596527, 0.018935075029730797, 1.2066316604614258, 0.0026964505668729544,
        -0.0038426220417022705, -0.05543806776404381, 0.016382435336709023, -0.3776109516620636,
        -0.07517704367637634, 1.2037315368652344, -0.03580886498093605, -0.07851681113243103,
        -0.011213400401175022
    ],
}


def _ease(alpha: float, ease: str = "cosine") -> float:
    a = float(alpha)
    if a <= 0.0:
        return 0.0
    if a >= 1.0:
        return 1.0
    e = str(ease or "cosine").lower()
    if e == "linear":
        return a
    # cosine ease-in-out
    return 0.5 - 0.5 * np.cos(np.pi * a)


def _parse_safe_idle_pose_ids(arg, presets_dict) -> list[int]:
    """
    解析 --safe_idle_pose_id。
    支持：
      - 单个数字：2
      - 逗号分隔列表："1,2"
    """
    if arg is None:
        ids = [0]
    elif isinstance(arg, int):
        ids = [int(arg)]
    else:
        s = str(arg).strip()
        if s == "":
            ids = [0]
        else:
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            ids = [int(p) for p in parts] if parts else [0]
    missing = [i for i in ids if i not in presets_dict]
    if missing:
        raise ValueError(f"--safe_idle_pose_id 包含不存在的 preset id：{missing}，可用：{sorted(presets_dict.keys())}")
    return ids


class _ActionRamp:
    """Smooth ramp between two action commands (body/neck/hands)."""

    def __init__(self) -> None:
        self.active = False
        self.t0 = 0.0
        self.seconds = 0.0
        self.ease = "cosine"
        self.from_body = np.zeros((35,), dtype=float)
        self.from_neck = np.zeros((2,), dtype=float)
        self.from_hand_l = np.zeros((7,), dtype=float)
        self.from_hand_r = np.zeros((7,), dtype=float)
        self.to_body = np.zeros((35,), dtype=float)
        self.to_neck = np.zeros((2,), dtype=float)
        self.to_hand_l = np.zeros((7,), dtype=float)
        self.to_hand_r = np.zeros((7,), dtype=float)
        # Optional multi-stage plan
        self._plan: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
        self._plan_idx: int = 0

    def start(
        self,
        from_body: np.ndarray,
        from_neck: np.ndarray,
        from_hand_l: np.ndarray,
        from_hand_r: np.ndarray,
        to_body: np.ndarray,
        to_neck: np.ndarray,
        to_hand_l: np.ndarray,
        to_hand_r: np.ndarray,
        seconds: float,
        ease: str,
    ) -> None:
        self._plan = []
        self._plan_idx = 0
        self.active = True
        self.t0 = time.time()
        self.seconds = max(0.0, float(seconds))
        self.ease = str(ease or "cosine")
        self.from_body = np.asarray(from_body, dtype=float).reshape(35).copy()
        self.from_neck = np.asarray(from_neck, dtype=float).reshape(2).copy()
        self.from_hand_l = np.asarray(from_hand_l, dtype=float).reshape(7).copy()
        self.from_hand_r = np.asarray(from_hand_r, dtype=float).reshape(7).copy()
        self.to_body = np.asarray(to_body, dtype=float).reshape(35).copy()
        self.to_neck = np.asarray(to_neck, dtype=float).reshape(2).copy()
        self.to_hand_l = np.asarray(to_hand_l, dtype=float).reshape(7).copy()
        self.to_hand_r = np.asarray(to_hand_r, dtype=float).reshape(7).copy()

    def start_plan(
        self,
        from_body: np.ndarray,
        from_neck: np.ndarray,
        from_hand_l: np.ndarray,
        from_hand_r: np.ndarray,
        stages: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
        ease: str,
    ) -> None:
        if not stages:
            self.active = False
            return
        self._plan = [(np.asarray(b, dtype=float).copy(),
                       np.asarray(n, dtype=float).copy(),
                       np.asarray(hl, dtype=float).copy(),
                       np.asarray(hr, dtype=float).copy(),
                       float(sec)) for (b, n, hl, hr, sec) in stages]
        self._plan_idx = 0
        self.active = True
        self.t0 = time.time()
        self.ease = str(ease or "cosine")
        self.from_body = np.asarray(from_body, dtype=float).reshape(35).copy()
        self.from_neck = np.asarray(from_neck, dtype=float).reshape(2).copy()
        self.from_hand_l = np.asarray(from_hand_l, dtype=float).reshape(7).copy()
        self.from_hand_r = np.asarray(from_hand_r, dtype=float).reshape(7).copy()
        b0, n0, hl0, hr0, sec0 = self._plan[0]
        self.to_body = np.asarray(b0, dtype=float).reshape(35).copy()
        self.to_neck = np.asarray(n0, dtype=float).reshape(2).copy()
        self.to_hand_l = np.asarray(hl0, dtype=float).reshape(7).copy()
        self.to_hand_r = np.asarray(hr0, dtype=float).reshape(7).copy()
        self.seconds = max(0.0, float(sec0))

    def value(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        if not self.active or self.seconds <= 1e-6:
            self.active = False
            return self.to_body, self.to_neck, self.to_hand_l, self.to_hand_r, True
        alpha = (time.time() - self.t0) / max(1e-6, float(self.seconds))
        w = _ease(alpha, ease=self.ease)
        body = self.from_body + w * (self.to_body - self.from_body)
        neck = self.from_neck + w * (self.to_neck - self.from_neck)
        hl = self.from_hand_l + w * (self.to_hand_l - self.from_hand_l)
        hr = self.from_hand_r + w * (self.to_hand_r - self.from_hand_r)
        done = float(alpha) >= 1.0 - 1e-6
        if done:
            if self._plan and (self._plan_idx < len(self._plan) - 1):
                # advance to next stage
                self._plan_idx += 1
                self.from_body, self.from_neck, self.from_hand_l, self.from_hand_r = (
                    self.to_body.copy(),
                    self.to_neck.copy(),
                    self.to_hand_l.copy(),
                    self.to_hand_r.copy(),
                )
                b, n, hl2, hr2, sec = self._plan[self._plan_idx]
                self.to_body = np.asarray(b, dtype=float).reshape(35).copy()
                self.to_neck = np.asarray(n, dtype=float).reshape(2).copy()
                self.to_hand_l = np.asarray(hl2, dtype=float).reshape(7).copy()
                self.to_hand_r = np.asarray(hr2, dtype=float).reshape(7).copy()
                self.seconds = max(0.0, float(sec))
                self.t0 = time.time()
                done = False
            else:
                self.active = False
        return body, neck, hl, hr, done

def start_interpolation(state_machine, start_obs, end_obs, duration=1.0):
    """Start interpolation from start_obs to end_obs over given duration"""
    state_machine.is_interpolating = True
    state_machine.interpolation_start_time = time.time()
    state_machine.interpolation_duration = duration
    state_machine.interpolation_start_obs = start_obs.copy() if start_obs is not None else None
    state_machine.interpolation_target_obs = end_obs.copy() if end_obs is not None else None
    

def get_interpolated_obs(state_machine):
    """Get current interpolated observation, returns None if interpolation complete"""
    if (not state_machine.is_interpolating or 
        state_machine.interpolation_start_obs is None or 
        state_machine.interpolation_target_obs is None or 
        state_machine.interpolation_start_time is None):
        return None
    elapsed_time = time.time() - state_machine.interpolation_start_time
    progress = min(elapsed_time / state_machine.interpolation_duration, 1.0)
    
    # Linear interpolation
    interp_obs = state_machine.interpolation_start_obs + (state_machine.interpolation_target_obs - state_machine.interpolation_start_obs) * progress
    
    # Check if interpolation is complete
    if progress >= 1.0:
        state_machine.is_interpolating = False
        return state_machine.interpolation_target_obs
    
    return interp_obs

def extract_mimic_obs_whole_body(qpos, last_qpos, dt=1/30):
    """Extract whole body mimic observations from robot joint positions (35 dims)"""
    root_pos, last_root_pos = qpos[0:3], last_qpos[0:3]
    root_quat, last_root_quat = qpos[3:7], last_qpos[3:7]
    robot_joints = qpos[7:].copy()  # Make a copy to avoid modifying original
    base_vel = (root_pos - last_root_pos) / dt
    base_ang_vel = quat_diff_np(last_root_quat, root_quat, scalar_first=True) / dt
    roll, pitch, yaw = euler_from_quaternion_np(root_quat.reshape(1, -1), scalar_first=True)
    # convert root vel to local frame
    base_vel_local = quat_rotate_inverse_np(root_quat, base_vel, scalar_first=True)
    base_ang_vel_local = quat_rotate_inverse_np(root_quat, base_ang_vel, scalar_first=True)
    
    # Standard mimic observation (35 dims)
    height = root_pos[2:3]
    # print("height: ", height)
    mimic_obs = np.concatenate([
        base_vel_local[:2],  # xy velocity (2 dims)
        root_pos[2:3],       # z position (1 dim)
        roll, pitch,         # roll, pitch (2 dims)
        base_ang_vel_local[2:3],  # yaw angular velocity (1 dim)
        robot_joints         # joint positions (29 dims)
    ])
    
    return mimic_obs



class StateMachine:
    def __init__(self, enable_smooth=False, smooth_window_size=5, use_pinch=False):
        """
        State process for teleoperation:
        idle -> teleop -> pause -> teleop ... -> idle -> exit
        """
        self.state = "teleop"
        self.previous_state = "idle"
        self.right_key_one_was_pressed = False
        self.left_key_one_was_pressed = False
        self.left_axis_click_was_pressed = False
        # Interpolation state
        self.is_interpolating = False
        self.interpolation_start_time = None
        self.interpolation_duration = 2.0  # seconds
        self.interpolation_start_obs = None
        self.interpolation_target_obs = None
        self.current_mimic_obs = None
        self.last_mimic_obs = None
        self.current_neck_data = None
        self.last_neck_data = None

        # Hand state - interpolation values (0.0 = open, 1.0 = closed)
        self.hand_left_position = 0.0  # 0.0 = fully open, 1.0 = fully closed
        self.hand_right_position = 0.0
        self.use_pinch = use_pinch
        # Hand control parameters
        self.hand_movement_step = 0.05  # 5% movement per press/hold
        
        # Velocity commands from joystick
        self.velocity_commands = np.array([0.0, 0.0, 0.0])  # [vx, vy, vyaw]
        
        # Smooth filtering
        self.enable_smooth = enable_smooth
        self.smooth_window_size = smooth_window_size
        self.smooth_history = []  # Store recent observations for sliding window

    def update(self, controller_data):
        """Update state machine with controller data"""
        # Store previous state
        self.previous_state = self.state
        
        # Get current button states
        right_key_current = controller_data.get('RightController', {}).get('key_one', False)
        left_key_current = controller_data.get('LeftController', {}).get('key_one', False)
        
        # Hand control - index_trig for close, grip for open
        right_index_trig_current = controller_data.get('RightController', {}).get('index_trig', False)
        left_index_trig_current = controller_data.get('LeftController', {}).get('index_trig', False)
        right_grip_current = controller_data.get('RightController', {}).get('grip', False)
        left_grip_current = controller_data.get('LeftController', {}).get('grip', False)

        # Emergency stop - left controller axis_click
        left_axis_click_current = controller_data.get('LeftController', {}).get('axis_click', False)

        # Detect button presses
        right_key_just_pressed = right_key_current and not self.right_key_one_was_pressed
        left_key_just_pressed = left_key_current and not self.left_key_one_was_pressed
        left_axis_click_just_pressed = left_axis_click_current and not self.left_axis_click_was_pressed

        # Handle left axis click - emergency stop
        if left_axis_click_just_pressed:
            self._emergency_stop()

        # Handle left key press - exit from any state
        if left_key_just_pressed:
            self.state = "exit"

        # Handle right key press - cycle between idle, teleop, pause
        elif right_key_just_pressed:
            if self.state == "idle":
                self.state = "teleop"
            elif self.state == "teleop":
                self.state = "pause"
            elif self.state == "pause":
                self.state = "teleop"

        # Handle hand control - continuous interpolation
        # Right hand control
        if right_index_trig_current:  # Close right hand
            new_position = min(1.0, self.hand_right_position + self.hand_movement_step)
            if new_position != self.hand_right_position:
                self.hand_right_position = new_position
                print(f"Right hand closing: {self.hand_right_position:.1f}")
        elif right_grip_current:  # Open right hand
            new_position = max(0.0, self.hand_right_position - self.hand_movement_step)
            if new_position != self.hand_right_position:
                self.hand_right_position = new_position
                print(f"Right hand opening: {self.hand_right_position:.1f}")
        
        # Left hand control
        if left_index_trig_current:  # Close left hand
            new_position = min(1.0, self.hand_left_position + self.hand_movement_step)
            if new_position != self.hand_left_position:
                self.hand_left_position = new_position
                print(f"Left hand closing: {self.hand_left_position:.1f}")
        elif left_grip_current:  # Open left hand
            new_position = max(0.0, self.hand_left_position - self.hand_movement_step)
            if new_position != self.hand_left_position:
                self.hand_left_position = new_position
                print(f"Left hand opening: {self.hand_left_position:.1f}")
        
        # Extract velocity commands from controller axes
        self._update_velocity_commands(controller_data)
        
        # Update button state tracking
        self.right_key_one_was_pressed = right_key_current
        self.left_key_one_was_pressed = left_key_current
        self.left_axis_click_was_pressed = left_axis_click_current
    
    def _update_velocity_commands(self, controller_data):
        """Update velocity commands from controller axes"""
        left_axis = controller_data.get('LeftController', {}).get('axis', [0.0, 0.0])
        right_axis = controller_data.get('RightController', {}).get('axis', [0.0, 0.0])
        
        # Use left stick for xy movement, right stick for yaw rotation
        if len(left_axis) >= 2 and len(right_axis) >= 2:
            # Scale factors for velocity commands
            xy_scale = 2.0  # m/s
            yaw_scale = 3.0  # rad/s
            
            self.velocity_commands[0] = left_axis[1] * xy_scale   # forward/backward (y axis inverted)
            self.velocity_commands[1] = -left_axis[0] * xy_scale  # left/right (x axis inverted)
            self.velocity_commands[2] = -right_axis[0] * yaw_scale  # yaw rotation (x axis inverted)
    
    def has_state_changed(self):
        """Check if state has changed since last update"""
        return self.state != self.previous_state
    
    
    def set_current_mimic_obs(self, mimic_obs):
        """Update current mimic obs"""
        self.current_mimic_obs = mimic_obs.copy() if mimic_obs is not None else None
        
    def set_last_mimic_obs(self, mimic_obs):
        """Update last mimic obs (used when entering pause)"""
        self.last_mimic_obs = mimic_obs.copy() if mimic_obs is not None else None
        
    def set_last_neck_data(self, neck_data):
        """Update last neck data (used when entering pause)"""
        self.last_neck_data = neck_data[:] if neck_data is not None else None
        
    def set_current_neck_data(self, neck_data):
        """Update current neck data"""
        self.current_neck_data = neck_data[:] if neck_data is not None else None
    
    def get_current_state(self):
        return self.state
    

    def get_velocity_commands(self):
        return self.velocity_commands.copy()
        
    def is_teleop_active(self):
        """Return True if currently in teleop state"""
        return self.state == "teleop"
        
    def should_exit(self):
        """Return True if should exit the program"""
        return self.state == "exit"
        
    def should_process_data(self):
        """Return True if should process motion data"""
        return self.state == "teleop" and not self.is_interpolating
    
    def get_hand_state(self):
        return self.hand_left_position, self.hand_right_position
    
    def get_hand_pose(self, robot_name):
        """Get interpolated hand poses based on current hand positions"""
        use_pinch = self.use_pinch
        # Get open and closed poses
        
        if not use_pinch:
            left_open = DEFAULT_HAND_POSE[robot_name]['left']['open']
            left_closed = DEFAULT_HAND_POSE[robot_name]['left']['close']
            right_open = DEFAULT_HAND_POSE[robot_name]['right']['open']
            right_closed = DEFAULT_HAND_POSE[robot_name]['right']['close']
        else:
            left_fully_open = DEFAULT_HAND_POSE[robot_name]['left']['open_pinch']
            left_fully_closed = DEFAULT_HAND_POSE[robot_name]['left']['close_pinch']
            right_fully_open = DEFAULT_HAND_POSE[robot_name]['right']['open_pinch']
            right_fully_closed = DEFAULT_HAND_POSE[robot_name]['right']['close_pinch']

            # compute the intermediate poses to shortern the distance betwen open and close
            # ratio * open + (1 - ratio) * closed
            ratio_open = 0.8
            ratio_closed = 0.0
            left_open =  left_fully_open * ratio_open + (1 - ratio_open) * left_fully_closed
            left_closed = left_fully_open * ratio_closed + (1 - ratio_closed) * left_fully_closed
            right_open = right_fully_open * ratio_open + (1 - ratio_open) * right_fully_closed
            right_closed = right_fully_open * ratio_closed + (1 - ratio_closed) * right_fully_closed
        
        # Interpolate between open and closed poses
        left_pose = left_open + (left_closed - left_open) * self.hand_left_position
        right_pose = right_open + (right_closed - right_open) * self.hand_right_position
        
        return left_pose, right_pose
    
    def apply_smooth(self, mimic_obs):
        """Apply sliding window smoothing to mimic observations"""
        if not self.enable_smooth or mimic_obs is None:
            return mimic_obs
            
        # Convert to numpy array if needed
        obs_array = np.array(mimic_obs) if not isinstance(mimic_obs, np.ndarray) else mimic_obs.copy()
        
        # Add current observation to history
        self.smooth_history.append(obs_array)
        
        # Keep only the recent window_size observations
        if len(self.smooth_history) > self.smooth_window_size:
            self.smooth_history.pop(0)
            
        # Apply sliding window average
        if len(self.smooth_history) >= 2:  # Need at least 2 observations for smoothing
            # Stack all observations in history
            history_stack = np.stack(self.smooth_history, axis=0)  # Shape: (history_len, obs_dim)
            # Compute mean across the time dimension
            smoothed_obs = np.mean(history_stack, axis=0)
            return smoothed_obs
        else:
            # Not enough history, return original observation
            return obs_array
    
    def reset_smooth_history(self):
        """Reset smooth history (call when transitioning states)"""
        self.smooth_history = []
    
    def _emergency_stop(self):
        """Emergency stop: kill sim2real.sh process (server_low_level_g1_real_future.py)"""
        try:
            print("[EMERGENCY STOP] Killing sim2real.sh process...")
            # Kill sim2real.sh which contains server_low_level_g1_real_future.py
            result = subprocess.run(['pkill', '-f', 'sim2real.sh'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("[EMERGENCY STOP] Successfully killed sim2real.sh process")
            else:
                print(f"[EMERGENCY STOP] pkill returned code {result.returncode}")

            # Also try to kill the specific server script directly as backup
            result2 = subprocess.run(['pkill', '-f', 'server_low_level_g1_real_future.py'], 
                                   capture_output=True, text=True, timeout=5)
            if result2.returncode == 0:
                print("[EMERGENCY STOP] Successfully killed server_low_level_g1_real_future.py process")
            else:
                print(f"[EMERGENCY STOP] pkill for server script returned code {result2.returncode}")
                
        except subprocess.TimeoutExpired:
            print("[EMERGENCY STOP] pkill command timed out")
        except Exception as e:
            print(f"[EMERGENCY STOP] Error executing pkill: {e}")

class XRobotTeleopToRobot:
    def __init__(self, args):
        self.args = args
        self.robot_name = args.robot
        self.xml_file = ROBOT_XML_DICT[args.robot]
        self.robot_base = ROBOT_BASE_DICT[args.robot]
        
        print(f"Pinch mode: {self.args.pinch_mode}")
        # Initialize state tracking
        self.last_qpos = None
        self.last_time = time.time()
        self.target_fps = args.target_fps
        self.measured_dt = 1/ self.target_fps # default fallback dt

        # Initialize components
        self.teleop_data_streamer = None
        self.redis_client = None
        self.retarget = None
        self.model = None
        self.data = None
        self.state_machine = StateMachine(
            enable_smooth=args.smooth,
            smooth_window_size=args.smooth_window_size,
            use_pinch=args.pinch_mode
        )
        self.rate = None
        
        # Video recording
        self.video_writer = None
        self.renderer = None
        self.video_filename = None
        self.frame_count = 0
        
        # FPS monitoring
        self.fps_monitor = FPSMonitor(
            enable_detailed_stats=args.measure_fps,
            quick_print_interval=100,
            detailed_print_interval=1000,
            expected_fps=self.target_fps,
            name="Teleop Loop"
        )
        
        # Hand tracking data storage (26D format)
        self.left_hand_tracking_dict = None
        self.right_hand_tracking_dict = None
        self.left_hand_is_active = False
        self.right_hand_is_active = False

        # Keyboard-controlled toggle for whether to publish teleop data to Redis
        self._send_enabled = True
        self._send_enabled_lock = threading.Lock()
        # Keyboard-controlled toggle for freezing teleop while holding the last commanded pose
        # When enabled: keep publishing the last action_* (body/hand/neck) to Redis, but stop updating hand_tracking_*.
        self._hold_position_enabled = False

        # Cached last actions for "hold position" mode
        self._cached_action_body_35 = np.array(DEFAULT_MIMIC_OBS[self.robot_name][:35], dtype=float)
        # Start from default hand poses (will be overwritten once we publish real teleop)
        _lh0, _rh0 = DEFAULT_HAND_POSE[self.robot_name]['left']['open'], DEFAULT_HAND_POSE[self.robot_name]['right']['open']
        self._cached_action_hand_left_7 = np.array(_lh0, dtype=float)
        self._cached_action_hand_right_7 = np.array(_rh0, dtype=float)
        self._cached_action_neck_2 = np.array([0.0, 0.0], dtype=float)

        # Wuji 模式控制（无需额外 JSON 文件）：
        # - follow: 正常跟随 hand_tracking（Wuji server 读取 tracking 并 retarget）
        # - hold: 保持当前位置（Wuji server 重复 last_qpos）
        # - default: 回零位（Wuji server 使用 zero_pose）
        self._wuji_robot_key = "unitree_g1_with_hands"
        self._redis_key_wuji_mode_left = f"wuji_hand_mode_left_{self._wuji_robot_key}"
        self._redis_key_wuji_mode_right = f"wuji_hand_mode_right_{self._wuji_robot_key}"

        self._keyboard_thread = None
        self._keyboard_stop = threading.Event()
        self._stdin_fd = None
        self._stdin_old_termios = None
        self._evdev_hotkeys = None

        # Smooth ramp for start/toggle/exit
        self._toggle_ramp = _ActionRamp()
        self._last_pub_body_35 = np.array(DEFAULT_MIMIC_OBS[self.robot_name][:35], dtype=float)
        self._last_pub_hand_left_7 = np.array(_lh0, dtype=float)
        self._last_pub_hand_right_7 = np.array(_rh0, dtype=float)
        self._last_pub_neck_2 = np.array([0.0, 0.0], dtype=float)

        # Safe-idle command (used when send_enabled=False), selectable by preset id
        self._safe_idle_pose_ids = _parse_safe_idle_pose_ids(getattr(self.args, "safe_idle_pose_id", "0"), SAFE_IDLE_BODY_35_PRESETS)
        self._safe_idle_body_seq_35: list[np.ndarray] = []
        for _pid in self._safe_idle_pose_ids:
            _v = SAFE_IDLE_BODY_35_PRESETS[_pid]
            if len(_v) != 35:
                raise ValueError(f"SAFE_IDLE_BODY_35_PRESETS[{_pid}] must have length 35, got {len(_v)}")
            self._safe_idle_body_seq_35.append(np.array(_v, dtype=float).reshape(35))
        # 最终稳定 idle：序列最后一个（兼容单个数字）
        self._safe_idle_body_35 = self._safe_idle_body_seq_35[-1].copy()
        self._safe_idle_hand_left_7 = np.zeros((7,), dtype=float)
        self._safe_idle_hand_right_7 = np.zeros((7,), dtype=float)
        self._safe_idle_neck_2 = np.zeros((2,), dtype=float)

        # Optional startup ramp:
        # DO NOT start it here, because cached_action_* is still DEFAULT_MIMIC_OBS at init time.
        # Instead, arm it and start when we receive the first valid teleop mimic_obs in the main loop.
        self._start_ramp_pending = float(getattr(self.args, "start_ramp_seconds", 0.0)) > 1e-6
        self._start_ramp_seconds = float(getattr(self.args, "start_ramp_seconds", 0.0))
        self._start_ramp_ease = str(getattr(self.args, "ramp_ease", "cosine"))


    def setup_teleop_data_streamer(self):
        """Initialize and start the teleop data streamer"""
        self.teleop_data_streamer = XRobotStreamer()
        print("Teleop data streamer initialized")
        
    def setup_redis_connection(self):
        """Setup Redis connection"""
        redis_ip = self.args.redis_ip
        # 使用 decode_responses=False 以保持与 server_wuji_hand_redis.py 一致
        self.redis_client = redis.Redis(host=redis_ip, port=6379, db=0, decode_responses=False)
        self.redis_pipeline = self.redis_client.pipeline()
        self.redis_client.ping()
        print("Redis connected successfully")

    def setup_retargeting_system(self):
        """Initialize the motion retargeting system"""
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=self.args.actual_human_height,
        )
        print("Retargeting system initialized")
    
    def setup_mujoco_simulation(self):
        """Setup MuJoCo model and data"""
        self.model = mj.MjModel.from_xml_path(str(self.xml_file))
        self.data = mj.MjData(self.model)
        print("MuJoCo simulation initialized")
        
    def setup_video_recording(self):
        """Setup video recording if requested"""
        if not self.args.record_video:
            return
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f'teleop_recording_{timestamp}.mp4'
        
        # Try different codecs for better compatibility
        # First try H264 (better quality), fallback to mp4v
        fourcc_options = [
            ('avc1', 'H264'),  # Better quality, more compatible
            ('mp4v', 'MPEG-4'),  # Fallback
            ('XVID', 'XVID'),  # Another fallback
        ]
        
        width, height = 640, 480
        fps = 30
        
        self.video_writer = None
        for codec, name in fourcc_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                if writer.isOpened():
                    self.video_writer = writer
                    print(f"[Video] Recording enabled: {video_filename} ({name} codec, {fps}fps, {width}x{height})")
                    break
                else:
                    writer.release()
            except:
                pass
        
        if self.video_writer is None:
            print("[Video] WARNING: Failed to initialize video writer with all codecs")
            return
            
        self.renderer = mj.Renderer(self.model, height=height, width=width)
        self.video_filename = video_filename
        self.frame_count = 0
        
    def setup_rate_limiter(self):
        """Setup rate limiter for consistent FPS"""
        self.rate = RateLimiter(frequency=self.target_fps, warn=False)
        print(f"Rate limiter setup for {self.target_fps} FPS")
        
    def get_teleop_data(self):
        """Get current teleop data from streamer"""
        if self.teleop_data_streamer is not None:
            return self.teleop_data_streamer.get_current_frame()
        return None, None, None, None, None
        
    def process_retargeting(self, smplx_data):
        """Process motion retargeting and return observations"""
        if smplx_data is None or self.retarget is None:
            return None, None
            
        # Measure dt between retarget calls
        current_time = time.time()
        self.measured_dt = current_time - self.last_time
        self.last_time = current_time
        
        # Retarget till convergence
        qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)
        
        # Create mimic obs from retargeting
        if self.last_qpos is not None:
            current_retarget_obs = extract_mimic_obs_whole_body(qpos, self.last_qpos, dt=self.measured_dt)
        else:
            current_retarget_obs = DEFAULT_MIMIC_OBS[self.robot_name]
        
        self.last_qpos = qpos.copy()
        return qpos, current_retarget_obs
        
    def update_visualization(self, qpos, smplx_data, viewer):
        """Update MuJoCo visualization"""
        if qpos is None:
            return
            
        # Clean custom geometry
        if hasattr(viewer, 'user_scn') and viewer.user_scn is not None:
            viewer.user_scn.ngeom = 0
            
        # Draw the task targets for reference
        if smplx_data is not None and self.retarget is not None:
            for robot_link, ik_data in self.retarget.ik_match_table1.items():
                body_name = ik_data[0]
                if body_name not in smplx_data:
                    continue
                draw_frame(
                    self.retarget.scaled_human_data[body_name][0] - self.retarget.ground,
                    R.from_quat(smplx_data[body_name][1]).as_matrix(),
                    viewer,
                    0.1,
                    orientation_correction=R.from_quat(ik_data[-1]),
                )
                
        # Update the simulation
        if qpos is not None:
            self.data.qpos[:] = qpos.copy()
            mj.mj_forward(self.model, self.data)
            
            # Camera follow the pelvis
            self._update_camera_position(viewer)
        
    def _update_camera_position(self, viewer):
        """Update camera to follow the robot"""
        FOLLOW_CAMERA = True
        if FOLLOW_CAMERA:
            robot_base_pos = self.data.xpos[self.model.body(self.robot_base).id]
            viewer.cam.lookat = robot_base_pos
            viewer.cam.distance = 3.0
            
    def handle_state_transitions(self, current_retarget_obs):
        """Handle state machine transitions and interpolations"""
        if not self.state_machine.has_state_changed():
            return
            
        current_state = self.state_machine.get_current_state()
        previous_state = self.state_machine.previous_state
        
        print(f"State changed: {previous_state} -> {current_state}")
        
        if current_state == "teleop":
            self._handle_enter_teleop(previous_state, current_retarget_obs)
        elif current_state == "pause":
            self._handle_enter_pause()
            
    def _handle_enter_teleop(self, previous_state, current_retarget_obs):
        """Handle entering teleop state"""
        if previous_state in ["idle", "pause"]:
            self.state_machine.reset_smooth_history()
            print("Reset smooth history on entering teleop")

        if previous_state == "idle":
            if current_retarget_obs is not None:
                default_obs = DEFAULT_MIMIC_OBS[self.robot_name]
                start_interpolation(self.state_machine, default_obs, current_retarget_obs[:35])
                print("Interpolating from default to teleop...")
        elif previous_state == "pause":
            if (current_retarget_obs is not None and
                self.state_machine.last_mimic_obs is not None):
                last_obs_35d = self.state_machine.last_mimic_obs[:35] if len(self.state_machine.last_mimic_obs) > 35 else self.state_machine.last_mimic_obs
                start_interpolation(self.state_machine, last_obs_35d, current_retarget_obs[:35])
                print("Interpolating from pause to teleop...")
    def _handle_enter_pause(self):
        """Handle entering pause state"""
        if self.state_machine.current_mimic_obs is not None:
            self.state_machine.set_last_mimic_obs(self.state_machine.current_mimic_obs)
            print("Entered pause mode, storing last obs")
        if self.state_machine.current_neck_data is not None:
            self.state_machine.set_last_neck_data(self.state_machine.current_neck_data)
            print("Entered pause mode, storing last neck data")
            
    def determine_mimic_obs_to_send(self, current_retarget_obs):
        """Determine which mimic observation to send based on current state"""
        current_state = self.state_machine.get_current_state()

        if current_state == "idle":
            obs = DEFAULT_MIMIC_OBS[self.robot_name]
        elif current_state == "pause":
            if self.state_machine.last_mimic_obs is not None:
                obs = self.state_machine.last_mimic_obs[:35] if len(self.state_machine.last_mimic_obs) > 35 else self.state_machine.last_mimic_obs
            else:
                obs = DEFAULT_MIMIC_OBS[self.robot_name]
        elif current_state == "teleop":
            obs = self._get_teleop_mimic_obs(current_retarget_obs)
            obs = self.state_machine.apply_smooth(obs)
        else:
            obs = DEFAULT_MIMIC_OBS[self.robot_name]

        return obs
        
    def _get_teleop_mimic_obs(self, current_retarget_obs):
        """Get mimic obs for teleop state, handling interpolation"""
        if self.state_machine.is_interpolating:
            interp_obs = get_interpolated_obs(self.state_machine)
            if interp_obs is not None:
                self.state_machine.set_current_mimic_obs(interp_obs)
                return interp_obs
            return DEFAULT_MIMIC_OBS[self.robot_name]

        if current_retarget_obs is not None:
            obs_35d = current_retarget_obs[:35] if len(current_retarget_obs) > 35 else current_retarget_obs
            self.state_machine.set_current_mimic_obs(obs_35d)
            return obs_35d

        return DEFAULT_MIMIC_OBS[self.robot_name]
    
    def determine_neck_data_to_send(self, smplx_data):
        """Determine which neck data to send based on current state"""
       
        current_state = self.state_machine.get_current_state()
        
        # In non-teleop states, send default neck position [0, 0]
        if current_state in ["idle"]:
            return [0.0, 0.0]
        
        if current_state == "pause":
            # return [0.0, 0.0]
            # use last neck data
            if self.state_machine.last_neck_data is not None:
                return self.state_machine.last_neck_data
            else:
                return [0.0, 0.0]
            
        # In teleop state, extract neck data from smplx_data
        elif current_state == "teleop" and smplx_data is not None:
            scale = self.args.neck_retarget_scale
            neck_yaw, neck_pitch = human_head_to_robot_neck(smplx_data)
            return [neck_yaw * scale, neck_pitch * scale]
        
        # Default fallback
        return [0.0, 0.0]
            
    def send_to_redis(self, mimic_obs, neck_data=None):
        """Send mimic observations to Redis"""

        if self.redis_client is None:
            return

        with self._send_enabled_lock:
            send_enabled = self._send_enabled
            hold_enabled = self._hold_position_enabled

        # 即使 send_enabled=False，也更新“缓存动作”（仅用于后续平滑回到跟踪，不会写入 Redis）
        try:
            if mimic_obs is not None:
                self._cached_action_body_35 = np.array(mimic_obs, dtype=float).reshape(35).copy()
            if neck_data is not None:
                self._cached_action_neck_2 = np.array(neck_data, dtype=float).reshape(2).copy()
        except Exception:
            pass

        # Optional toggle ramp has highest priority: it smooths transitions for k/p.
        if getattr(self._toggle_ramp, "active", False):
            body35, neck2, hl7, hr7, _done = self._toggle_ramp.value()
            self.redis_pipeline.set("action_body_unitree_g1_with_hands", json.dumps(body35.tolist()))
            self.redis_pipeline.set("action_hand_left_unitree_g1_with_hands", json.dumps(hl7.tolist()))
            self.redis_pipeline.set("action_hand_right_unitree_g1_with_hands", json.dumps(hr7.tolist()))
            self.redis_pipeline.set("action_neck_unitree_g1_with_hands", json.dumps(neck2.tolist()))
            self.redis_pipeline.set("t_action", int(time.time() * 1000))
            self.redis_pipeline.execute()
            self._last_pub_body_35, self._last_pub_neck_2 = body35.copy(), neck2.copy()
            self._last_pub_hand_left_7, self._last_pub_hand_right_7 = hl7.copy(), hr7.copy()
            return

        # If sending is disabled, publish a safe idle action to avoid executing stale commands.
        if not send_enabled:
            self.redis_pipeline.set("action_body_unitree_g1_with_hands", json.dumps(self._safe_idle_body_35.tolist()))
            self.redis_pipeline.set("action_hand_left_unitree_g1_with_hands", json.dumps(self._safe_idle_hand_left_7.tolist()))
            self.redis_pipeline.set("action_hand_right_unitree_g1_with_hands", json.dumps(self._safe_idle_hand_right_7.tolist()))
            self.redis_pipeline.set("action_neck_unitree_g1_with_hands", json.dumps(self._safe_idle_neck_2.tolist()))
            self.redis_pipeline.set("t_action", int(time.time() * 1000))
            self.redis_pipeline.execute()
            self._last_pub_body_35, self._last_pub_neck_2 = self._safe_idle_body_35.copy(), self._safe_idle_neck_2.copy()
            self._last_pub_hand_left_7, self._last_pub_hand_right_7 = self._safe_idle_hand_left_7.copy(), self._safe_idle_hand_right_7.copy()
            return

        # If hold-position is enabled, keep publishing the last cached action to hold the current pose.
        if hold_enabled:
            self.redis_pipeline.set("action_body_unitree_g1_with_hands", json.dumps(self._cached_action_body_35.tolist()))
            self.redis_pipeline.set("action_hand_left_unitree_g1_with_hands", json.dumps(self._cached_action_hand_left_7.tolist()))
            self.redis_pipeline.set("action_hand_right_unitree_g1_with_hands", json.dumps(self._cached_action_hand_right_7.tolist()))
            self.redis_pipeline.set("action_neck_unitree_g1_with_hands", json.dumps(self._cached_action_neck_2.tolist()))
            self.redis_pipeline.set("t_action", int(time.time() * 1000))
            self.redis_pipeline.execute()
            self._last_pub_body_35, self._last_pub_neck_2 = self._cached_action_body_35.copy(), self._cached_action_neck_2.copy()
            self._last_pub_hand_left_7, self._last_pub_hand_right_7 = self._cached_action_hand_left_7.copy(), self._cached_action_hand_right_7.copy()
            return

        if mimic_obs is not None:
            # Expect 35D mimic observations
            assert len(mimic_obs) == 35, f"Expected 35 mimic obs dims, got {len(mimic_obs)}"
            self.redis_pipeline.set("action_body_unitree_g1_with_hands", json.dumps(mimic_obs.tolist()))

        # Send hand action to redis (7D joint positions for Unitree G1)
        hand_left_pose, hand_right_pose = self.state_machine.get_hand_pose(self.robot_name)
        self.redis_pipeline.set("action_hand_left_unitree_g1_with_hands", json.dumps(hand_left_pose.tolist()))
        self.redis_pipeline.set("action_hand_right_unitree_g1_with_hands", json.dumps(hand_right_pose.tolist()))

        # Send neck data to redis
        if neck_data is not None:
            self.redis_pipeline.set("action_neck_unitree_g1_with_hands", json.dumps(neck_data))

        # Send timestamp to redis
        self.redis_pipeline.set("t_action", int(time.time() * 1000))
        self.redis_pipeline.execute()

        # Track last published action (for ramp start)
        try:
            if mimic_obs is not None:
                self._last_pub_body_35 = np.array(mimic_obs, dtype=float).copy()
            self._last_pub_hand_left_7 = np.array(hand_left_pose, dtype=float).copy()
            self._last_pub_hand_right_7 = np.array(hand_right_pose, dtype=float).copy()
            if neck_data is not None:
                self._last_pub_neck_2 = np.array(neck_data, dtype=float).copy()
        except Exception:
            pass

        # Update cached actions for hold-position mode
        try:
            if mimic_obs is not None:
                self._cached_action_body_35 = np.array(mimic_obs, dtype=float).copy()
            self._cached_action_hand_left_7 = np.array(hand_left_pose, dtype=float).copy()
            self._cached_action_hand_right_7 = np.array(hand_right_pose, dtype=float).copy()
            if neck_data is not None:
                self._cached_action_neck_2 = np.array(neck_data, dtype=float).copy()
        except Exception:
            # Cache failures should not break the teleop loop
            pass

    
    def send_controller_data_to_redis(self, controller_data):
        """Send controller data to Redis"""
        if self.redis_client is None or controller_data is None:
            return
        with self._send_enabled_lock:
            if not self._send_enabled:
                return
        self.redis_client.set("controller_data", json.dumps(controller_data))
    
    def send_hand_tracking_data_to_redis(self):
        """Send hand tracking data (26D format) to Redis for Wuji hand retargeting"""
        if self.redis_client is None:
            return

        with self._send_enabled_lock:
            send_enabled = self._send_enabled
            hold_enabled = self._hold_position_enabled

        current_time_ms = int(time.time() * 1000)  # 当前时间戳（毫秒）

        # 写入 Wuji 模式（k/p 与全身一致）
        # - k (send_enabled=False) => default
        # - p (hold_enabled=True)  => hold
        # - else                  => follow
        mode = "follow"
        if not send_enabled:
            mode = "default"
        elif hold_enabled:
            mode = "hold"
        try:
            self.redis_client.set(self._redis_key_wuji_mode_left, mode)
            self.redis_client.set(self._redis_key_wuji_mode_right, mode)
        except Exception:
            pass

        # 当处于 hold/default 时，不再需要更新 tracking（Wuji server 自己根据 mode 处理）
        if mode in ["hold", "default"]:
            self.redis_client.set("hand_tracking_left_unitree_g1_with_hands", json.dumps({"is_active": False, "timestamp": current_time_ms}))
            self.redis_client.set("hand_tracking_right_unitree_g1_with_hands", json.dumps({"is_active": False, "timestamp": current_time_ms}))
            return

        # Send left hand tracking data (26D dictionary format)
        if self.left_hand_tracking_dict is not None:
            hand_tracking_data_left = {
                "is_active": self.left_hand_is_active,
                "timestamp": current_time_ms,
                **self.left_hand_tracking_dict
            }
            self.redis_client.set("hand_tracking_left_unitree_g1_with_hands", json.dumps(hand_tracking_data_left))
        else:
            self.redis_client.set("hand_tracking_left_unitree_g1_with_hands", json.dumps({"is_active": False, "timestamp": current_time_ms}))

        # Send right hand tracking data (26D dictionary format)
        if self.right_hand_tracking_dict is not None:
            hand_tracking_data_right = {
                "is_active": self.right_hand_is_active,
                "timestamp": current_time_ms,
                **self.right_hand_tracking_dict
            }
            self.redis_client.set("hand_tracking_right_unitree_g1_with_hands", json.dumps(hand_tracking_data_right))
        else:
            self.redis_client.set("hand_tracking_right_unitree_g1_with_hands", json.dumps({"is_active": False, "timestamp": current_time_ms}))
            
            
    def _get_current_published_actions(self):
        """Best-effort: current action being published (handles ramp-in-progress)."""
        if getattr(self._toggle_ramp, "active", False):
            body, neck, hl, hr, _ = self._toggle_ramp.value()
            return body, neck, hl, hr
        return self._last_pub_body_35, self._last_pub_neck_2, self._last_pub_hand_left_7, self._last_pub_hand_right_7

    def _start_toggle_ramp_if_needed(self, target_mode: str):
        secs = float(getattr(self.args, "toggle_ramp_seconds", 0.0))
        if secs <= 1e-6:
            return
        ease = str(getattr(self.args, "ramp_ease", "cosine"))
        from_body, from_neck, from_hl, from_hr = self._get_current_published_actions()

        if target_mode == "default":
            to_body, to_neck, to_hl, to_hr = (
                self._safe_idle_body_35,
                self._safe_idle_neck_2,
                self._safe_idle_hand_left_7,
                self._safe_idle_hand_right_7,
            )
        elif target_mode == "hold":
            to_body, to_neck, to_hl, to_hr = (
                self._cached_action_body_35,
                self._cached_action_neck_2,
                self._cached_action_hand_left_7,
                self._cached_action_hand_right_7,
            )
        else:  # follow
            to_body, to_neck, to_hl, to_hr = (
                self._cached_action_body_35,
                self._cached_action_neck_2,
                self._cached_action_hand_left_7,
                self._cached_action_hand_right_7,
            )

        self._toggle_ramp.start(
            from_body=from_body,
            from_neck=from_neck,
            from_hand_l=from_hl,
            from_hand_r=from_hr,
            to_body=to_body,
            to_neck=to_neck,
            to_hand_l=to_hl,
            to_hand_r=to_hr,
            seconds=secs,
            ease=ease,
        )

    def _start_send_toggle_ramp(self, enabled: bool) -> None:
        """k 切换 send_enabled 的平滑：支持 safe idle 序列（如 1,2）。"""
        secs_total = float(getattr(self.args, "toggle_ramp_seconds", 0.0))
        if secs_total <= 1e-6:
            return
        ease = str(getattr(self.args, "ramp_ease", "cosine"))
        from_body, from_neck, from_hl, from_hr = self._get_current_published_actions()

        # entering default (send_enabled=False): 1 -> 2 -> ...
        if not bool(enabled):
            if hasattr(self, "_safe_idle_body_seq_35") and len(getattr(self, "_safe_idle_body_seq_35", [])) >= 2:
                seq: list[np.ndarray] = list(getattr(self, "_safe_idle_body_seq_35"))
                per = secs_total / float(max(1, len(seq)))
                stages = []
                for b in seq:
                    stages.append((np.asarray(b, dtype=float).reshape(35),
                                   self._safe_idle_neck_2,
                                   self._safe_idle_hand_left_7,
                                   self._safe_idle_hand_right_7,
                                   float(per)))
                self._toggle_ramp.start_plan(
                    from_body=from_body,
                    from_neck=from_neck,
                    from_hand_l=from_hl,
                    from_hand_r=from_hr,
                    stages=stages,
                    ease=ease,
                )
            else:
                self._toggle_ramp.start(
                    from_body=from_body,
                    from_neck=from_neck,
                    from_hand_l=from_hl,
                    from_hand_r=from_hr,
                    to_body=self._safe_idle_body_35,
                    to_neck=self._safe_idle_neck_2,
                    to_hand_l=self._safe_idle_hand_left_7,
                    to_hand_r=self._safe_idle_hand_right_7,
                    seconds=secs_total,
                    ease=ease,
                )
            return

        # leaving default (send_enabled=True): ...2 -> 1 -> follow
        if hasattr(self, "_safe_idle_body_seq_35") and len(getattr(self, "_safe_idle_body_seq_35", [])) >= 2:
            seq: list[np.ndarray] = list(getattr(self, "_safe_idle_body_seq_35"))
            b1 = np.asarray(seq[-2], dtype=float).reshape(35)
            per = secs_total / 2.0
            stages = [
                (b1, self._safe_idle_neck_2, self._safe_idle_hand_left_7, self._safe_idle_hand_right_7, float(per)),
                (self._cached_action_body_35, self._cached_action_neck_2, self._cached_action_hand_left_7, self._cached_action_hand_right_7, float(per)),
            ]
            self._toggle_ramp.start_plan(
                from_body=from_body,
                from_neck=from_neck,
                from_hand_l=from_hl,
                from_hand_r=from_hr,
                stages=stages,
                ease=ease,
            )
        else:
            self._toggle_ramp.start(
                from_body=from_body,
                from_neck=from_neck,
                from_hand_l=from_hl,
                from_hand_r=from_hr,
                to_body=self._cached_action_body_35,
                to_neck=self._cached_action_neck_2,
                to_hand_l=self._cached_action_hand_left_7,
                to_hand_r=self._cached_action_hand_right_7,
                seconds=secs_total,
                ease=ease,
            )

    def _handle_hotkey(self, ch: str):
        toggle_key = getattr(self.args, "toggle_send_key", "k") or "k"
        toggle_key = str(toggle_key)[0]
        hold_key = getattr(self.args, "hold_position_key", "p") or "p"
        hold_key = str(hold_key)[0]

        if ch == toggle_key:
            with self._send_enabled_lock:
                self._send_enabled = not self._send_enabled
                enabled = self._send_enabled
                if not self._send_enabled:
                    self._hold_position_enabled = False
            self._start_send_toggle_ramp(bool(enabled))
            print(f"[Keyboard] send_enabled => {enabled}")
        elif ch == hold_key:
            with self._send_enabled_lock:
                if not self._send_enabled:
                    self._hold_position_enabled = False
                    enabled = self._hold_position_enabled
                else:
                    self._hold_position_enabled = not self._hold_position_enabled
                    enabled = self._hold_position_enabled
            self._start_toggle_ramp_if_needed("hold" if enabled else "follow")
            print(f"[Keyboard] hold_position_enabled => {enabled}")

    def _keyboard_loop(self):
        """Background loop: terminal key toggles whether we publish teleop data to Redis."""
        toggle_key = getattr(self.args, "toggle_send_key", "k") or "k"
        toggle_key = str(toggle_key)[0]
        hold_key = getattr(self.args, "hold_position_key", "p") or "p"
        hold_key = str(hold_key)[0]

        try:
            fd = sys.stdin.fileno()
            self._stdin_fd = fd
            self._stdin_old_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)

            print(f"[Keyboard] 按 '{toggle_key}' 切换 teleop 是否发送数据到 Redis（避开录制按键 r/q）")
            print(f"[Keyboard] 按 '{hold_key}' 切换“保持当前位置”(冻结 action，停止 hand_tracking 更新)")
            while not self._keyboard_stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == toggle_key or ch == hold_key:
                    self._handle_hotkey(ch)
        except Exception as e:
            print(f"[Keyboard] 键盘监听不可用：{e}")
        finally:
            try:
                if self._stdin_fd is not None and self._stdin_old_termios is not None:
                    termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_termios)
            except Exception:
                pass

    def _start_keyboard_toggle(self):
        if not getattr(self.args, "keyboard_toggle_send", False):
            return
        if self._keyboard_thread is not None:
            return
        backend = str(getattr(self.args, "keyboard_backend", "terminal") or "terminal").lower()
        if backend in ["evdev", "both"]:
            cfg = EvdevHotkeyConfig(
                device=str(getattr(self.args, "evdev_device", "auto")),
                grab=bool(getattr(self.args, "evdev_grab", False)),
            )
            self._evdev_hotkeys = EvdevHotkeys(cfg, callback=lambda c: self._handle_hotkey(str(c)[0]))
            self._evdev_hotkeys.start()
            print(f"[Keyboard] backend=evdev device={cfg.device} grab={cfg.grab}")
            print(f"[Keyboard] 按 '{str(getattr(self.args, 'toggle_send_key', 'k'))[0]}' 切换 send；按 '{str(getattr(self.args, 'hold_position_key', 'p'))[0]}' 切换 hold")
            if backend == "evdev":
                return
        self._keyboard_stop.clear()
        self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._keyboard_thread.start()

    def _stop_keyboard_toggle(self):
        self._keyboard_stop.set()
        try:
            if self._evdev_hotkeys is not None:
                self._evdev_hotkeys.stop()
        except Exception:
            pass
        self._evdev_hotkeys = None
        try:
            if self._keyboard_thread is not None and self._keyboard_thread.is_alive():
                self._keyboard_thread.join(timeout=0.5)
        except Exception:
            pass
        self._keyboard_thread = None

    def record_video_frame(self, viewer):
        """Record current frame to video if recording is enabled"""
        if not self.args.record_video or self.renderer is None or self.video_writer is None:
            return
            
        try:
            self.renderer.update_scene(self.data, camera=viewer.cam)
            pixels = self.renderer.render()
            
            # Convert from RGB to BGR (OpenCV uses BGR)
            frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)
            self.frame_count += 1
            
            # Print progress every 100 frames
            if self.frame_count % 100 == 0:
                print(f"[Video] Recorded {self.frame_count} frames", end='\r')
        except Exception as e:
            print(f"[Video] Error recording frame: {e}")
        
    def handle_exit_sequence(self, viewer):
        """Handle graceful exit with interpolation to default pose"""
        if self.state_machine.current_mimic_obs is not None:
            default_obs = DEFAULT_MIMIC_OBS[self.robot_name]
            current_obs = self.state_machine.current_mimic_obs[:35] if len(self.state_machine.current_mimic_obs) > 35 else self.state_machine.current_mimic_obs
            start_interpolation(self.state_machine, current_obs, default_obs)
            print("Interpolating to default pose before exit...")
            
            # Wait for interpolation to complete
            while self.state_machine.is_interpolating:
                interp_obs = get_interpolated_obs(self.state_machine)
                if interp_obs is not None:
                    # During exit sequence, send default neck position [0, 0]
                    neck_data_to_send = self.determine_neck_data_to_send(None)
                    self.send_to_redis(interp_obs, neck_data_to_send)
                viewer.sync()
                self.rate.sleep()
        
        # Release video writer if recording
        if self.video_writer is not None:
            self.video_writer.release()
            if hasattr(self, 'frame_count') and hasattr(self, 'video_filename'):
                duration = self.frame_count / 30.0  # 30 fps
                print(f"\n[Video] Recording saved: {self.video_filename}")
                print(f"[Video] Total frames: {self.frame_count}, Duration: {duration:.2f}s")
            else:
                print("\n[Video] Video recording saved")
                


    def initialize_all_systems(self):
        """Initialize all required systems"""
        print("Initializing teleop systems...")
        self.setup_teleop_data_streamer()
        self.setup_redis_connection()
        self.setup_retargeting_system()
        self.setup_mujoco_simulation()
        self.setup_video_recording()
        self.setup_rate_limiter()

        print("Teleop state machine initialized. Controls:")
        print("- Right controller key_one: Cycle through idle -> teleop -> pause -> teleop...")
        print("- Left controller key_one: Exit program")
        print("- Left controller axis_click: Emergency stop - kills sim2real.sh process")
        print("- Left controller axis: Control root xy velocity")
        print("- Right controller axis: Control yaw velocity")
        print("- Publishes 35-dimensional mimic observations")
        print(f"Starting in state: {self.state_machine.get_current_state()}")

        if self.state_machine.enable_smooth:
            print(f"- Smooth filtering: ENABLED (window size: {self.state_machine.smooth_window_size} frames)")
        else:
            print("- Smooth filtering: DISABLED")
        
        if self.fps_monitor.enable_detailed_stats:
            print(f"- FPS measurement: ENABLED (detailed stats every {self.fps_monitor.detailed_print_interval} steps)")
        else:
            print(f"- FPS measurement: Quick stats only (every {self.fps_monitor.quick_print_interval} steps)")

        print("Ready to receive teleop data.")

    def run(self):
        """
        Main execution loop
        1. 获取数据 → 2. 更新状态机 → 3. 处理重定向 → 4. 发送观察值 → 5. 可视化 → 6. 控制频率
        """
        self.initialize_all_systems()
        self._start_keyboard_toggle()
        
        try:
            # Start the viewer
            with mjv.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False
            ) as viewer:
                viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1

                while viewer.is_running():
                    # Get current teleop data
                    smplx_data, left_hand_data, right_hand_data, controller_data, headset_data = self.get_teleop_data()

                    # 解包手部追踪数据 (left_hand_data 是元组: (is_active, hand_data_dict))
                    if left_hand_data and isinstance(left_hand_data, tuple) and len(left_hand_data) == 2:
                        self.left_hand_is_active, self.left_hand_tracking_dict = left_hand_data
                    else:
                        self.left_hand_is_active = False
                        self.left_hand_tracking_dict = None

                    if right_hand_data and isinstance(right_hand_data, tuple) and len(right_hand_data) == 2:
                        self.right_hand_is_active, self.right_hand_tracking_dict = right_hand_data
                    else:
                        self.right_hand_is_active = False
                        self.right_hand_tracking_dict = None

                    # Send hand tracking data to Redis
                    self.send_hand_tracking_data_to_redis()

                    # Update state machine
                    if controller_data is not None:
                        self.state_machine.update(controller_data)
                        self.send_controller_data_to_redis(controller_data)

                    # Check if we should exit
                    if self.state_machine.should_exit():
                        print("Exit requested via controller")
                        self.handle_exit_sequence(viewer)
                        break

                    # Process retargeting if we have data
                    qpos, current_retarget_obs = None, None
                    if smplx_data is not None:
                        qpos, current_retarget_obs = self.process_retargeting(smplx_data)
                        self.update_visualization(qpos, smplx_data, viewer)

                    # Handle state transitions
                    self.handle_state_transitions(current_retarget_obs)

                    # Determine and send mimic observations
                    mimic_obs_to_send = self.determine_mimic_obs_to_send(current_retarget_obs)
                    neck_data_to_send = self.determine_neck_data_to_send(smplx_data)

                    # Store current neck data in state machine for pause state handling
                    if neck_data_to_send is not None:
                        self.state_machine.set_current_neck_data(neck_data_to_send)

                    # Startup ramp: safe idle -> first valid teleop output (only once).
                    # Gate it by send_enabled/hold_enabled to avoid ramp overriding k-disabled mode.
                    if getattr(self, "_start_ramp_pending", False) and mimic_obs_to_send is not None:
                        try:
                            with self._send_enabled_lock:
                                _send_enabled = bool(self._send_enabled)
                                _hold_enabled = bool(self._hold_position_enabled)
                            if _send_enabled and (not _hold_enabled):
                                obs35 = np.asarray(mimic_obs_to_send, dtype=float).reshape(35)
                                neck2 = np.asarray(neck_data_to_send if neck_data_to_send is not None else [0.0, 0.0], dtype=float).reshape(2)
                                hl7, hr7 = self.state_machine.get_hand_pose(self.robot_name)
                                self._toggle_ramp.start(
                                    from_body=self._safe_idle_body_35,
                                    from_neck=self._safe_idle_neck_2,
                                    from_hand_l=self._safe_idle_hand_left_7,
                                    from_hand_r=self._safe_idle_hand_right_7,
                                    to_body=obs35,
                                    to_neck=neck2,
                                    to_hand_l=np.asarray(hl7, dtype=float).reshape(7),
                                    to_hand_r=np.asarray(hr7, dtype=float).reshape(7),
                                    seconds=float(getattr(self, "_start_ramp_seconds", 0.0)),
                                    ease=str(getattr(self, "_start_ramp_ease", "cosine")),
                                )
                                self._start_ramp_pending = False
                        except Exception:
                            # Don't break the teleop loop due to ramp issues.
                            self._start_ramp_pending = False

                    self.send_to_redis(mimic_obs_to_send, neck_data_to_send)

                    # Update visualization and record video
                    viewer.sync()
                    self.record_video_frame(viewer)

                    # FPS monitoring
                    self.fps_monitor.tick()

                    self.rate.sleep()
        finally:
            # Optional exit ramp: smoothly go to safe idle before stopping (best-effort).
            try:
                secs = float(getattr(self.args, "exit_ramp_seconds", 0.0))
                if secs > 1e-6 and self.redis_client is not None:
                    from_body, from_neck, from_hl, from_hr = self._get_current_published_actions()
                    self._toggle_ramp.start(
                        from_body=from_body,
                        from_neck=from_neck,
                        from_hand_l=from_hl,
                        from_hand_r=from_hr,
                        to_body=self._safe_idle_body_35,
                        to_neck=self._safe_idle_neck_2,
                        to_hand_l=self._safe_idle_hand_left_7,
                        to_hand_r=self._safe_idle_hand_right_7,
                        seconds=secs,
                        ease=str(getattr(self.args, "ramp_ease", "cosine")),
                    )
                    t0 = time.time()
                    while self._toggle_ramp.active and (time.time() - t0) < (secs + 0.5):
                        self.send_to_redis(mimic_obs=None, neck_data=None)
                        time.sleep(min(0.02, secs / 50.0))
            except Exception:
                pass
            self._stop_keyboard_toggle()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands"],
        default="unitree_g1",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Whether to record the video.",
    )
    parser.add_argument(
        "--pinch_mode",
        action="store_true",
        help="Whether to use pinch mode for hand control.",
        default=False,
    )
    parser.add_argument(
        "--redis_ip",
        type=str,
        default="localhost",
        help="Redis IP",
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.5,
        help="Actual human height for retargeting.",
    )   
    parser.add_argument(
        "--neck_retarget_scale",
        type=float,
        default=1.5,
        help="Scale factor for neck data.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Enable smooth filtering for mimic observations in teleop mode.",
    )
    parser.add_argument(
        "--smooth_window_size",
        type=int,
        default=5,
        help="Window size for sliding window smoothing (default: 5 frames).",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=100,
        help="Target FPS for the teleop system.",
    )
    parser.add_argument(
        "--measure_fps",
        type=int,
        default=0,
        help="Measure and print detailed FPS statistics (0=disabled, 1=enabled).",
    )
    parser.add_argument(
        "--safe_idle_pose_id",
        type=str,
        default="0",
        help=(
            "When k disables sending (send_enabled=False), publish this safe idle pose preset id (35D mimic_obs). "
            "支持单个数字（如 2）或序列（如 1,2）。当传序列时："
            "按 k 进入 default：先平滑到 1，再平滑到 2；再按 k 回到跟踪：先 2->1，再 1->正常跟踪。"
        ),
    )
    parser.add_argument(
        "--keyboard_toggle_send",
        action="store_true",
        help="Enable terminal keyboard toggle for whether to publish teleop data to Redis.",
    )
    parser.add_argument(
        "--keyboard_backend",
        type=str,
        default="terminal",
        choices=["terminal", "evdev", "both"],
        help="Hotkey backend: terminal (stdin, requires focus) / evdev (Linux /dev/input, works with footswitch) / both (evdev + terminal).",
    )
    parser.add_argument(
        "--evdev_device",
        type=str,
        default="auto",
        help="evdev 设备路径，如 /dev/input/event3 或 /dev/input/by-id/...；auto 尝试自动选择",
    )
    parser.add_argument(
        "--evdev_grab",
        action="store_true",
        help="evdev 是否 grab 设备（避免其他程序同时收到按键；谨慎使用）",
    )
    parser.add_argument(
        "--toggle_send_key",
        type=str,
        default="k",
        help="Single key to toggle sending (default: 'k'; avoid recorder keys like r/q).",
    )
    parser.add_argument(
        "--hold_position_key",
        type=str,
        default="p",
        help="Single key to toggle holding current pose (default: 'p').",
    )
    parser.add_argument("--start_ramp_seconds", type=float, default=0.0, help="启动时从 safe idle 平滑过渡到当前姿态（秒，0 关闭）")
    parser.add_argument("--toggle_ramp_seconds", type=float, default=0.0, help="k/p 切换时的平滑插值时长（秒，0 关闭）")
    parser.add_argument("--exit_ramp_seconds", type=float, default=0.0, help="退出时平滑回到 safe idle（秒，0 关闭）")
    parser.add_argument("--ramp_ease", type=str, default="cosine", choices=["linear", "cosine"], help="插值曲线（默认 cosine）")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    teleop_robot = XRobotTeleopToRobot(args)
    teleop_robot.run()