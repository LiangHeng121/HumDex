#!/usr/bin/env python3
"""
TWIST2 Real Robot Controller with Wuji Hand

基于 server_low_level_g1_real.py，将手部控制替换为 Wuji 手控制。
从 Redis 读取手部追踪数据（26维），转换为21维 MediaPipe 格式，
使用 WujiHandRetargeter 重定向后控制 Wuji 灵巧手。
"""
import argparse
import random
import time
import json
import numpy as np
import torch
import redis
from collections import deque
import sys
from pathlib import Path

from robot_control.g1_wrapper import G1RealWorldEnv
from robot_control.config import Config
import os
from data_utils.rot_utils import quatToEuler

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import wujihandpy
except ImportError:
    print("❌ 错误: 未安装 wujihandpy，请先安装:")
    print("   pip install wujihandpy")
    wujihandpy = None

# 添加 wuji_retargeting 到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WUJI_RETARGETING_PATH = PROJECT_ROOT / "wuji_retargeting"
if str(WUJI_RETARGETING_PATH) not in sys.path:
    sys.path.insert(0, str(WUJI_RETARGETING_PATH))

try:
    from wuji_retargeting import WujiHandRetargeter
    from wuji_retargeting.mediapipe import apply_mediapipe_transformations
except ImportError as e:
    print(f"⚠️  警告: 无法导入 wuji_retargeting: {e}")
    print("   Wuji 手控制将被禁用")
    WujiHandRetargeter = None
    apply_mediapipe_transformations = None


# 26维手部关节名称（与 xrobot_utils.py 中的定义一致）
HAND_JOINT_NAMES_26 = [
    "Wrist", "Palm",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip", 
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"
]

# 26维到21维 MediaPipe 格式的映射索引
# MediaPipe 格式: [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
# 26维格式: [Wrist, Palm, Thumb(4), Index(5), Middle(5), Ring(5), Pinky(5)]
MEDIAPIPE_MAPPING_26_TO_21 = [
    1,   # 0: Palm -> Wrist (使用 Palm 作为 Wrist)
    2,   # 1: ThumbMetacarpal -> Thumb CMC
    3,   # 2: ThumbProximal -> Thumb MCP
    4,   # 3: ThumbDistal -> Thumb IP
    5,   # 4: ThumbTip -> Thumb Tip
    6,   # 5: IndexMetacarpal -> Index MCP
    7,   # 6: IndexProximal -> Index PIP
    8,   # 7: IndexIntermediate -> Index DIP
    10,  # 8: IndexTip -> Index Tip (跳过 IndexDistal)
    11,  # 9: MiddleMetacarpal -> Middle MCP
    12,  # 10: MiddleProximal -> Middle PIP
    13,  # 11: MiddleIntermediate -> Middle DIP
    15,  # 12: MiddleTip -> Middle Tip (跳过 MiddleDistal)
    16,  # 13: RingMetacarpal -> Ring MCP
    17,  # 14: RingProximal -> Ring PIP
    18,  # 15: RingIntermediate -> Ring DIP
    20,  # 16: RingTip -> Ring Tip (跳过 RingDistal)
    21,  # 17: LittleMetacarpal -> Pinky MCP
    22,  # 18: LittleProximal -> Pinky PIP
    23,  # 19: LittleIntermediate -> Pinky DIP
    25,  # 20: LittleTip -> Pinky Tip (跳过 LittleDistal)
]


def hand_26d_to_mediapipe_21d(hand_data_dict, hand_side="left"):
    """
    将26维手部追踪数据转换为21维 MediaPipe 格式
    
    Args:
        hand_data_dict: 字典，包含26个关节的数据
                      格式: {"LeftHandWrist": [[x,y,z], [qw,qx,qy,qz]], ...}
        hand_side: "left" 或 "right"
    
    Returns:
        numpy array of shape (21, 3) - MediaPipe 格式的手部关键点
    """
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"
    
    # 提取26个关节的位置
    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_data_dict:
            pos = hand_data_dict[key][0]  # [x, y, z]
            joint_positions_26[i] = pos
        else:
            # 如果缺少数据，使用零值
            joint_positions_26[i] = [0.0, 0.0, 0.0]
    
    # 使用映射索引转换为21维
    mediapipe_21d = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21]
    
    # 将腕部坐标设为0（作为原点）
    wrist_pos = mediapipe_21d[0].copy()  # 保存原始腕部位置
    mediapipe_21d = mediapipe_21d - wrist_pos  # 所有点相对于腕部
    
    # 其他坐标（除了腕部）乘以缩放因子
    scale_factor = 1.2
    mediapipe_21d[1:] = mediapipe_21d[1:] * scale_factor  # 索引1-20都乘以缩放因子
    # 腕部保持为0（索引0）
    
    return mediapipe_21d


def smooth_move_wuji(hand, controller, target_qpos, duration=0.02, steps=5):
    """
    平滑移动到某个 5×4 的关节目标（用于 Wuji 手）
    
    Args:
        hand: wujihandpy.Hand 对象
        controller: wujihandpy 控制器对象
        target_qpos: numpy array of shape (5, 4)
        duration: 平滑移动持续时间（秒）
        steps: 平滑移动步数
    """
    target_qpos = target_qpos.reshape(5, 4)
    try:
        cur = controller.get_joint_actual_position()
    except:
        cur = np.zeros((5, 4), dtype=np.float32)
    
    for t in np.linspace(0, 1, steps):
        q = cur * (1 - t) + target_qpos * t
        controller.set_joint_target_position(q)
        time.sleep(duration / steps)


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


class EMASmoother:
    """Exponential Moving Average smoother for body actions."""
    
    def __init__(self, alpha=0.1, initial_value=None):
        """
        Args:
            alpha: Smoothing factor (0.0=no smoothing, 1.0=maximum smoothing)
            initial_value: Initial value for smoothing (if None, will use first input)
        """
        self.alpha = alpha
        self.initialized = False
        self.smoothed_value = initial_value
        
    def smooth(self, new_value):
        """Apply EMA smoothing to new value."""
        if not self.initialized:
            self.smoothed_value = new_value.copy() if hasattr(new_value, 'copy') else new_value
            self.initialized = True
            return self.smoothed_value
        
        # EMA formula: smoothed = alpha * new + (1 - alpha) * previous
        self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value
    
    def reset(self):
        """Reset the smoother to uninitialized state."""
        self.initialized = False
        self.smoothed_value = None


class WujiHandController:
    """Wuji 手控制器，从 Redis 读取手部追踪数据并控制 Wuji 手"""
    
    def __init__(self, redis_client, hand_side="left", smooth_enabled=True, smooth_steps=5):
        """
        Args:
            redis_client: Redis 客户端对象
            hand_side: "left" 或 "right"
            smooth_enabled: 是否启用平滑移动
            smooth_steps: 平滑移动步数
        """
        self.hand_side = hand_side.lower()
        assert self.hand_side in ["left", "right"], "hand_side must be 'left' or 'right'"
        
        self.redis_client = redis_client
        self.redis_key_hand_tracking = f"hand_tracking_{self.hand_side}_unitree_g1_with_hands"
        self.smooth_enabled = smooth_enabled
        self.smooth_steps = smooth_steps
        
        # 初始化 Wuji 手
        self.hand = None
        self.controller = None
        self.zero_pose = None
        self.retargeter = None
        self.last_qpos = None
        
        if wujihandpy is None:
            print(f"⚠️  Wuji {self.hand_side} 手: wujihandpy 未安装，手部控制将被禁用")
            return
        
        if WujiHandRetargeter is None:
            print(f"⚠️  Wuji {self.hand_side} 手: wuji_retargeting 未安装，手部控制将被禁用")
            return
        
        try:
            print(f"🤖 初始化 Wuji {self.hand_side} 手...")
            self.hand = wujihandpy.Hand()
            self.hand.write_joint_enabled(True)
            self.controller = self.hand.realtime_controller(
                enable_upstream=True,
                filter=wujihandpy.filter.LowPass(cutoff_freq=10.0)
            )
            time.sleep(0.4)
            
            # 获取零位
            self.zero_pose = self.hand.get_joint_actual_position()
            self.last_qpos = self.zero_pose.copy()
            print(f"✅ Wuji {self.hand_side} 手初始化完成")
            
            # 初始化重定向器
            print(f"🔄 初始化 WujiHandRetargeter ({self.hand_side})...")
            self.retargeter = WujiHandRetargeter(hand_side=self.hand_side)
            print("✅ 重定向器初始化完成")
            
        except (RuntimeError, TimeoutError) as e:
            print(f"⚠️  Wuji {self.hand_side} 手初始化失败: {e}")
            print("   手部控制将被禁用")
            self.hand = None
            self.controller = None
            self.retargeter = None
    
    def get_hand_tracking_data_from_redis(self):
        """
        从 Redis 读取手部追踪数据（26维字典格式）
        
        Returns:
            tuple: (is_active, hand_data_dict) 或 (None, None)
        """
        try:
            data = self.redis_client.get(self.redis_key_hand_tracking)
            
            if data is None:
                return None, None
            
            # 解析 JSON
            hand_data = json.loads(data)
            
            # 检查数据格式
            if isinstance(hand_data, dict):
                # 检查数据是否新鲜（通过时间戳）
                data_timestamp = hand_data.get("timestamp", 0)
                current_time_ms = int(time.time() * 1000)
                time_diff_ms = current_time_ms - data_timestamp
                
                # 如果时间差超过 500ms，认为数据过期
                if time_diff_ms > 500:
                    return None, None
                
                # 检查 is_active 标志
                is_active = hand_data.get("is_active", False)
                if not is_active:
                    return None, None
                
                # 提取手部数据（排除元数据）
                hand_dict = {k: v for k, v in hand_data.items() 
                           if k not in ["is_active", "timestamp"]}
                
                return is_active, hand_dict
            else:
                return None, None
                
        except Exception as e:
            # 静默处理错误，避免频繁打印
            return None, None
    
    def update(self):
        """
        更新 Wuji 手控制（从 Redis 读取数据并控制）
        
        Returns:
            bool: 是否成功更新
        """
        if self.hand is None or self.controller is None or self.retargeter is None:
            return False
        
        # 从 Redis 读取手部追踪数据
        is_active, hand_data_dict = self.get_hand_tracking_data_from_redis()
        # print(f"is_active: {is_active}, hand_data_dict: {hand_data_dict}")
        
        if not is_active or hand_data_dict is None:
            return False
        
        # 1. 将26维转换为21维 MediaPipe 格式
        mediapipe_21d = hand_26d_to_mediapipe_21d(hand_data_dict, self.hand_side)
        
        # 2. 应用 MediaPipe 变换
        mediapipe_transformed = apply_mediapipe_transformations(
            mediapipe_21d, 
            hand_type=self.hand_side
        )
        
        # 3. 使用 WujiHandRetargeter 进行重定向
        retarget_result = self.retargeter.retarget(mediapipe_transformed)
        wuji_20d = retarget_result.robot_qpos.reshape(5, 4)
        
        # 4. 控制 Wuji 手
        if self.smooth_enabled:
            smooth_move_wuji(self.hand, self.controller, wuji_20d, 
                           duration=0.02, steps=self.smooth_steps)
        else:
            self.controller.set_joint_target_position(wuji_20d)
        
        self.last_qpos = wuji_20d.copy()
        return True
    
    def cleanup(self):
        """清理资源"""
        if self.hand is None or self.controller is None:
            return
        
        # 平滑回到零位
        if self.zero_pose is not None:
            smooth_move_wuji(self.hand, self.controller, self.zero_pose, duration=1.0, steps=50)
        self.controller.close()
        self.hand.write_joint_enabled(False)
        print(f"✅ Wuji {self.hand_side} 手已关闭")


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
    providers = []
    available = ort.get_available_providers()
    if device.startswith('cuda'):
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        else:
            print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
    providers.append('CPUExecutionProvider')
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_name)


class RealTimePolicyController(object):
    """
    Real robot controller for TWIST2 policy with Wuji hand control.
    基于 server_low_level_g1_real.py，手部控制替换为 Wuji 手。
    """
    def __init__(self, 
                 policy_path,
                 config_path,
                 device='cuda',
                 net='eno1',
                 use_wuji_hand=False,
                 wuji_hand_sides=["left", "right"],
                 record_proprio=False,
                 smooth_body=0.0,
                 wuji_hand_smooth=True,
                 wuji_hand_smooth_steps=5):
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_pipeline = self.redis_client.pipeline()
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
            exit()
       
        self.config = Config(config_path)
        self.env = G1RealWorldEnv(net=net, config=self.config)
        
        # Wuji 手控制（延迟初始化，避免在 reset_robot() 等待期间占用 USB 资源）
        self.use_wuji_hand = use_wuji_hand
        self.wuji_hand_sides = wuji_hand_sides
        self.wuji_hand_smooth = wuji_hand_smooth
        self.wuji_hand_smooth_steps = wuji_hand_smooth_steps
        self.wuji_hand_controllers = {}
        # 延迟初始化，在 reset_robot() 之后进行

        self.device = device
        self.policy = load_onnx_policy(policy_path, device)

        self.num_actions = 29
        self.default_dof_pos = self.config.default_angles
        
        # scaling factors
        self.ang_vel_scale = 0.25
        self.dof_vel_scale = 0.05
        self.dof_pos_scale = 1.0
        self.ankle_idx = [4, 5, 10, 11]

        # TWIST2 observation structure
        self.n_mimic_obs = 35        # 6 + 29 (modified: root_vel_xy + root_pos_z + roll_pitch + yaw_ang_vel + dof_pos)
        self.n_proprio = 92          # from config analysis  
        self.n_obs_single = 127      # n_mimic_obs + n_proprio = 35 + 92 = 127
        self.history_len = 10
        
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs  # 127*11 + 35 = 1402
        
        print(f"TWIST2 Real Controller Configuration:")
        print(f"  n_mimic_obs: {self.n_mimic_obs}")
        print(f"  n_proprio: {self.n_proprio}")
        print(f"  n_obs_single: {self.n_obs_single}")
        print(f"  history_len: {self.history_len}")
        print(f"  total_obs_size: {self.total_obs_size}")
        print(f"  Use Wuji hand: {self.use_wuji_hand}")

        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        self.control_dt = self.config.control_dt
        self.action_scale = self.config.action_scale
        
        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None
        
        # Smoothing processing
        self.smooth_body = smooth_body
        if smooth_body > 0.0:
            self.body_smoother = EMASmoother(alpha=smooth_body)
            print(f"Body action smoothing enabled with alpha={smooth_body}")
        else:
            self.body_smoother = None

        
    def reset_robot(self):
        print("Press START on remote to move to default position ...")
        self.env.move_to_default_pos()

        print("Now in default position, press A to continue ...")
        self.env.default_pos_state()

        print("Robot will hold default pos. If needed, do other checks here.")
        
        # 在 reset_robot() 完成后初始化 Wuji 手控制器
        # 这样可以避免在等待期间占用 USB 资源，防止缓冲区溢出
        if self.use_wuji_hand and len(self.wuji_hand_controllers) == 0:
            print("🤖 初始化 Wuji 手控制器...")
            # time.sleep(2.0)  # 等待 2 秒
            for hand_side in self.wuji_hand_sides:
                self.wuji_hand_controllers[hand_side] = WujiHandController(
                    redis_client=self.redis_client,
                    hand_side=hand_side,
                    smooth_enabled=self.wuji_hand_smooth,
                    smooth_steps=self.wuji_hand_smooth_steps
                )
            print("✅ Wuji 手控制器初始化完成")

    def run(self):
        self.reset_robot()
        print("Begin main TWIST2 policy loop. Press [Select] on remote to exit.")

        try:
            while True:
                t_start = time.time()

                # Send remote control signals to Redis for motion server
                if self.redis_client:
                    # Send B button status (for motion start)
                    b_pressed = self.env.read_controller_input().keys == self.env.controller_mapping["B"]
                    self.redis_client.set("motion_start_signal", "1" if b_pressed else "0")
                    
                    # Send Select button status (for motion exit)
                    select_pressed = self.env.read_controller_input().keys == self.env.controller_mapping["select"]
                    self.redis_client.set("motion_exit_signal", "1" if select_pressed else "0")
                    
                if self.env.read_controller_input().keys == self.env.controller_mapping["select"]:
                    print("Select pressed, exiting main loop.")
                    break
                
                dof_pos, dof_vel, quat, ang_vel, dof_temp, dof_tau, dof_vol = self.env.get_robot_state()
                
                rpy = quatToEuler(quat)

                obs_dof_vel = dof_vel.copy()
                obs_dof_vel[self.ankle_idx] = 0.0

                obs_proprio = np.concatenate([
                    ang_vel * self.ang_vel_scale,
                    rpy[:2], # 只使用 roll 和 pitch
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    obs_dof_vel * self.dof_vel_scale,
                    self.last_action
                ])
                
                state_body = np.concatenate([
                    ang_vel,
                    rpy[:2],
                    dof_pos]) # 3+2+29 = 34 dims

                self.redis_pipeline.set("state_body_unitree_g1_with_hands", json.dumps(state_body.tolist()))
                
                # 不再发送手部状态到 Redis（因为使用 Wuji 手，不需要 Unitree 手状态）
                # 如果需要，可以发送零状态
                self.redis_pipeline.set("state_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
                self.redis_pipeline.set("state_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
                
                # execute the pipeline once here for setting the keys
                self.redis_pipeline.execute()

                # 从 Redis 接收模仿观察
                keys = ["action_body_unitree_g1_with_hands", "action_hand_left_unitree_g1_with_hands", "action_hand_right_unitree_g1_with_hands", "action_neck_unitree_g1_with_hands"]
                for key in keys:
                    self.redis_pipeline.get(key)
                redis_results = self.redis_pipeline.execute()
                action_mimic = json.loads(redis_results[0])
                action_hand_left = json.loads(redis_results[1])
                action_hand_right = json.loads(redis_results[2])
                action_neck = json.loads(redis_results[3])

                if action_mimic is None:
                    print("action_mimic is None")
                
                # Apply smoothing to body actions if enabled
                if self.body_smoother is not None:
                    action_mimic = self.body_smoother.smooth(np.array(action_mimic, dtype=np.float32))
                    action_mimic = action_mimic.tolist()
            
                obs_full = np.concatenate([action_mimic, obs_proprio])
                
                obs_hist = np.array(self.proprio_history_buf).flatten()
                self.proprio_history_buf.append(obs_full)
                
                future_obs = action_mimic.copy()
                
                obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
                
                assert obs_buf.shape[0] == self.total_obs_size, f"Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}"
                
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                
                self.last_action = raw_action.copy()

                raw_action = np.clip(raw_action, -10.0, 10.0)
                target_dof_pos = self.default_dof_pos + raw_action * self.action_scale

                kp_scale = 1.0
                kd_scale = 1.0
                self.env.send_robot_action(target_dof_pos, kp_scale, kd_scale)
                
                # 更新 Wuji 手控制
                if self.use_wuji_hand:
                    for hand_side, controller in self.wuji_hand_controllers.items():
                        controller.update()
                
                elapsed = time.time() - t_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)

                if self.record_proprio:
                    proprio_data = {
                        'timestamp': time.time(),
                        'body_dof_pos': dof_pos.tolist(),
                        'target_dof_pos': action_mimic.tolist()[-29:],
                        'temperature': dof_temp.tolist(),
                        'tau': dof_tau.tolist(),
                        'voltage': dof_vol.tolist(),
                    }
                    self.proprio_recordings.append(proprio_data)
                

        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.record_proprio and self.proprio_recordings:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'logs/twist2_real_recordings_{timestamp}.json'
                os.makedirs('logs', exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(self.proprio_recordings, f)
                print(f"Proprioceptive recordings saved as {filename}")

            # 清理 Wuji 手
            if self.use_wuji_hand:
                for controller in self.wuji_hand_controllers.values():
                    controller.cleanup()

            self.env.close()
            print("TWIST2 real controller with Wuji hand finished.")


def main():
    parser = argparse.ArgumentParser(description='Run TWIST2 policy on real G1 robot with Wuji hand')
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to TWIST2 ONNX policy file')
    parser.add_argument('--config', type=str, default="robot_control/configs/g1.yaml",
                        help='Path to robot configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run policy on (cuda/cpu)')
    parser.add_argument('--net', type=str, default='wlp0s20f3',
                        help='Network interface for robot communication')
    parser.add_argument('--use_wuji_hand', action='store_true',
                        help='Enable Wuji hand control')
    parser.add_argument('--wuji_hand_sides', type=str, nargs='+', default=['left', 'right'],
                        choices=['left', 'right'],
                        help='Which Wuji hands to control (default: left right)')
    parser.add_argument('--record_proprio', action='store_true',
                        help='Record proprioceptive data')
    parser.add_argument('--smooth_body', type=float, default=0.0,
                        help='Smoothing factor for body actions (0.0=no smoothing, 1.0=maximum smoothing)')
    parser.add_argument('--wuji_hand_smooth', action='store_true', default=True,
                        help='Enable smoothing for Wuji hand control')
    parser.add_argument('--wuji_hand_smooth_steps', type=int, default=5,
                        help='Number of steps for Wuji hand smoothing')
    
    args = parser.parse_args()

    
    # 验证文件存在
    if not os.path.exists(args.policy):
        print(f"Error: Policy file {args.policy} does not exist")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} does not exist")
        return
    
    print(f"Starting TWIST2 real robot controller with Wuji hand...")
    print(f"  Policy file: {args.policy}")
    print(f"  Config file: {args.config}")
    print(f"  Device: {args.device}")
    print(f"  Network interface: {args.net}")
    print(f"  Use Wuji hand: {args.use_wuji_hand}")
    if args.use_wuji_hand:
        print(f"  Wuji hand sides: {args.wuji_hand_sides}")
        print(f"  Wuji hand smooth: {args.wuji_hand_smooth}")
        print(f"  Wuji hand smooth steps: {args.wuji_hand_smooth_steps}")
    print(f"  Record proprio: {args.record_proprio}")
    print(f"  Smooth body: {args.smooth_body}")
    
    # 安全提示
    print("\n" + "="*50)
    print("SAFETY WARNING:")
    print("You are about to run a policy on a real robot.")
    print("Make sure the robot is in a safe environment.")
    print("Press Ctrl+C to stop at any time.")
    print("Use the remote controller [Select] button to exit.")
    print("="*50 + "\n")
    
    controller = RealTimePolicyController(
        policy_path=args.policy,
        config_path=args.config,
        device=args.device,
        net=args.net,
        use_wuji_hand=args.use_wuji_hand,
        wuji_hand_sides=args.wuji_hand_sides,
        record_proprio=args.record_proprio,
        smooth_body=args.smooth_body,
        wuji_hand_smooth=args.wuji_hand_smooth,
        wuji_hand_smooth_steps=args.wuji_hand_smooth_steps,
    )
    
    controller.run()
    


if __name__ == "__main__":
    main()

