"""Visualization for humanoid body (35D) and hand (20D) actions."""

import numpy as np
import mujoco
import torch
from collections import deque
from pathlib import Path
from typing import List, Optional

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import imageio
except ImportError:
    imageio = None


def save_video(frames: List[np.ndarray], path: str, fps: int = 30) -> bool:
    if imageio is None:
        print("Cannot save video: pip install imageio imageio-ffmpeg")
        return False
    if not frames:
        return False
    try:
        imageio.mimsave(path, frames, fps=fps)
        print(f"Video saved: {path} ({len(frames)} frames @ {fps} FPS)")
        return True
    except Exception as e:
        print(f"Video save failed: {path}: {e}")
        return False


def save_frame(frame: np.ndarray, path: str) -> bool:
    if imageio is None:
        return False
    imageio.imwrite(path, frame)
    return True


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to roll-pitch-yaw."""
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([roll, pitch, yaw])


class OnnxPolicyWrapper:
    """ONNX Runtime wrapper for low-level RL policy."""

    def __init__(self, session, input_name: str, output_index: int = 0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        return torch.from_numpy(outputs[self.output_index].astype(np.float32))


def load_onnx_policy(policy_path: str, device: str = 'cpu') -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required: pip install onnxruntime")

    providers = ['CPUExecutionProvider']
    if device.startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')

    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX policy loaded: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_name)


class HumanoidVisualizer:
    """Visualize 35D body actions via low-level RL policy + MuJoCo sim.

    Takes high-level 35D action commands, uses an ONNX RL policy to produce
    joint torques, and renders the resulting motion.
    """

    def __init__(self, xml_path: str, policy_path: str,
                 device: str = 'cpu', width: int = 640, height: int = 480):
        print(f"Loading MuJoCo model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)

        print(f"Loading RL policy: {policy_path}")
        self.policy = load_onnx_policy(policy_path, device)
        self.device = device

        # 29-DOF robot configuration
        self.default_dof_pos = np.array([
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,             # left leg
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,             # right leg
            0.0, 0.0, 0.0,                                # torso
            0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,          # left arm
            0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,         # right arm
        ])

        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),                       # base position
            np.array([1, 0, 0, 0]),                        # base quaternion
            np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0,
                0.0, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0,
            ])
        ])

        self.stiffness = np.array([
            100, 100, 100, 150, 40, 40,
            100, 100, 100, 150, 40, 40,
            150, 150, 150,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
        ])
        self.damping = np.array([
            2, 2, 2, 4, 2, 2,
            2, 2, 2, 4, 2, 2,
            4, 4, 4,
            5, 5, 5, 5, 0.2, 0.2, 0.2,
            5, 5, 5, 5, 0.2, 0.2, 0.2,
        ])
        self.torque_limits = np.array([
            100, 100, 100, 150, 40, 40,
            100, 100, 100, 150, 40, 40,
            150, 150, 150,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
        ])
        self.action_scale = np.full(29, 0.5)
        self.ankle_idx = [4, 5, 10, 11]
        self.last_action = np.zeros(29, dtype=np.float32)

        # Observation buffer: proprioception (92D) + mimic target (35D) + history
        self.n_mimic_obs = 35
        self.n_proprio = 92
        self.n_obs_single = 127
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs

        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

        self._reset()

        print(f"Creating offscreen renderer ({width}x{height})")
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.sim_dt = 0.001
        self.control_decimation = 20  # 50Hz control

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[:] = self.mujoco_default_dof_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.proprio_history_buf.clear()
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

    def step(self, action_35d: np.ndarray) -> np.ndarray:
        """Execute one 50Hz control step. Returns rendered frame (H, W, 3)."""
        dof_pos = self.data.qpos[7:36].copy()
        dof_vel = self.data.qvel[6:35].copy()
        quat = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()

        rpy = quat_to_euler(quat)
        obs_dof_vel = dof_vel.copy()
        obs_dof_vel[self.ankle_idx] = 0.0

        obs_proprio = np.concatenate([
            ang_vel * 0.25, rpy[:2],
            dof_pos - self.default_dof_pos,
            obs_dof_vel * 0.05,
            self.last_action
        ])

        obs_full = np.concatenate([action_35d, obs_proprio])
        obs_hist = np.array(self.proprio_history_buf).flatten()
        self.proprio_history_buf.append(obs_full)
        obs_buf = np.concatenate([obs_full, obs_hist, action_35d])

        obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0)
        with torch.no_grad():
            raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()

        self.last_action = raw_action.copy()
        raw_action = np.clip(raw_action, -10.0, 10.0)
        pd_target = raw_action * self.action_scale + self.default_dof_pos

        for _ in range(self.control_decimation):
            dof_pos = self.data.qpos[7:36].copy()
            dof_vel = self.data.qvel[6:35].copy()
            torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)

        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def visualize(self, actions: np.ndarray, output_video: Optional[str] = None,
                  fps: int = 30, warmup_steps: int = 100, reset: bool = True,
                  verbose: bool = True) -> List[np.ndarray]:
        """Run actions through sim and collect rendered frames."""
        if reset:
            self._reset()

        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[1] != 35:
            raise ValueError(f"Expected (T, 35) actions, got {actions.shape}")

        if verbose:
            print(f"Visualizing {len(actions)} actions...")

        if warmup_steps > 0:
            for _ in range(warmup_steps):
                self.step(actions[0])

        frames = []
        for i, action in enumerate(actions):
            frames.append(self.step(action))
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(actions)} frames")

        if output_video:
            save_video(frames, output_video, fps=fps)

        return frames


class HandVisualizer:
    """Visualize 20D dexterous hand actions via direct joint control in MuJoCo."""

    def __init__(self, xml_path: str, hand_side: str = "left",
                 width: int = 640, height: int = 480):
        self.hand_side = hand_side
        self.width = width
        self.height = height

        print(f"Loading Wuji {hand_side} hand model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self._set_neutral_ctrl()
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        print(f"Creating offscreen renderer ({width}x{height})")
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = 180
        self.camera.elevation = -20
        self.camera.distance = 0.5
        self.camera.lookat[:] = [0, 0, 0.05]

        self.sim_dt = self.model.opt.timestep

    def _set_neutral_ctrl(self):
        for i in range(self.model.nu):
            if self.model.actuator_ctrllimited[i]:
                r = self.model.actuator_ctrlrange[i]
                self.data.ctrl[i] = (r[0] + r[1]) / 2
            else:
                self.data.ctrl[i] = 0.0

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self._set_neutral_ctrl()
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    def step(self, action_20d: np.ndarray) -> np.ndarray:
        """Execute one step with 20D (or 5x4) hand action. Returns frame (H, W, 3)."""
        if isinstance(action_20d, np.ndarray) and action_20d.shape == (5, 4):
            action_20d = action_20d.flatten()

        action_flat = np.array(action_20d).flatten()
        if len(action_flat) != 20:
            raise ValueError(f"Expected 20D action, got {len(action_flat)}D")

        min_len = min(len(action_flat), self.model.nu)
        self.data.ctrl[:min_len] = action_flat[:min_len]
        mujoco.mj_step(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def visualize(self, actions: np.ndarray, output_video: Optional[str] = None,
                  fps: int = 30, warmup_steps: int = 50, reset: bool = True,
                  verbose: bool = True) -> List[np.ndarray]:
        """Run hand actions through sim and collect rendered frames."""
        if reset:
            self._reset()

        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        elif actions.ndim == 3:
            actions = actions.reshape(len(actions), -1)
        elif actions.ndim == 2 and actions.shape[1] != 20:
            raise ValueError(f"Expected (T, 20) actions, got {actions.shape}")

        if verbose:
            print(f"Visualizing {len(actions)} hand actions...")

        if warmup_steps > 0:
            for _ in range(warmup_steps):
                self.step(actions[0])

        frames = []
        for i, action in enumerate(actions):
            frames.append(self.step(action))
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(actions)} frames")

        if output_video:
            save_video(frames, output_video, fps=fps)

        return frames


def get_default_paths() -> dict:
    """Default asset paths relative to this module."""
    d = Path(__file__).parent
    return {
        "body_xml": str(d / "assets/g1/g1_sim2sim_29dof.xml"),
        "body_policy": str(d / "assets/ckpts/twist2_1017_20k.onnx"),
        "left_hand_xml": str(d / "assets/wuji_hand/left.xml"),
        "right_hand_xml": str(d / "assets/wuji_hand/right.xml"),
    }
