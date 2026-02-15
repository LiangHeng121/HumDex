# TWIST2 代码学习路径指南

## 📚 系统架构概览

TWIST2采用**分层控制架构**，主要分为三个层次：

```
┌─────────────────────────────────────────┐
│  高层：运动重定向 (GMR)                  │
│  - PICO VR → 人体动作 → 机器人关节空间   │
└──────────────┬──────────────────────────┘
               │ Redis通信
┌──────────────▼──────────────────────────┐
│  中层：运动服务器 (Motion Server)        │
│  - 离线运动流 / 在线遥操作               │
└──────────────┬──────────────────────────┘
               │ Redis通信 (mimic_obs)
┌──────────────▼──────────────────────────┐
│  低层：RL控制器 (Policy Controller)      │
│  - ONNX模型推理 → 关节力矩控制           │
└─────────────────────────────────────────┘
```

---

## 🎯 学习路径规划

### 阶段一：基础理解（1-2天）

#### 1.1 从入口脚本开始
**目标**：理解系统如何启动和运行

**推荐阅读顺序**：
1. `sim2sim.sh` - 仿真环境启动脚本
   - 理解如何加载ONNX策略
   - 了解MuJoCo仿真环境设置
   
2. `sim2real.sh` - 真实机器人启动脚本
   - 理解网络配置
   - 了解真实机器人接口

3. `teleop.sh` - 遥操作启动脚本
   - 理解PICO VR数据流
   - 了解运动重定向入口

4. `run_motion_server.sh` - 运动服务器脚本
   - 理解离线运动流播放

**关键文件**：
- `deploy_real/server_low_level_g1_sim.py` (仿真控制器)
- `deploy_real/server_low_level_g1_real.py` (真实控制器)
- `deploy_real/xrobot_teleop_to_robot_w_hand.py` (遥操作主程序)

---

### 阶段二：数据流理解（2-3天）

#### 2.1 观察空间 (Observation Space)
**目标**：理解RL策略的输入格式

**关键文件**：
- `deploy_real/data_utils/params.py` - 默认观察值定义
  - `DEFAULT_MIMIC_OBS`: 35维模仿观察（root_vel_xy + root_pos_z + roll_pitch + yaw_ang_vel + 29个关节位置）
  
- `deploy_real/server_low_level_g1_real.py` (131-144行)
  ```python
  self.n_mimic_obs = 35        # 模仿观察维度
  self.n_proprio = 92          # 本体感受观察维度
  self.n_obs_single = 127      # 单帧观察 = 35 + 92
  self.history_len = 10        # 历史长度
  self.total_obs_size = 1402   # 总观察维度 = 127*11 + 35
  ```

**观察空间组成**：
- **Mimic Obs (35维)**：来自高层运动重定向的目标状态
  - root_vel_xy (2) + root_pos_z (1) + roll_pitch (2) + yaw_ang_vel (1) + dof_pos (29)
- **Proprio Obs (92维)**：机器人当前状态
  - 位置、速度、接触等信息
- **History (10帧)**：历史状态用于时序建模

#### 2.2 动作空间 (Action Space)
**目标**：理解RL策略的输出格式

**关键文件**：
- `deploy_real/server_low_level_g1_real.py` (122行)
  ```python
  self.num_actions = 29  # 29个关节的目标位置
  ```

**动作处理**：
- 动作是关节位置增量
- 通过PD控制器转换为力矩
- 有动作平滑处理（EMA）

#### 2.3 Redis通信协议
**目标**：理解高层和低层如何通信

**关键文件**：
- `deploy_real/server_motion_lib.py` - 运动服务器
  - 发送 `mimic_obs` 到Redis
  - 发送 `neck_data` 到Redis
  
- `deploy_real/server_low_level_g1_real.py` - 低层控制器
  - 从Redis读取 `mimic_obs`
  - 从Redis读取 `neck_data`

**Redis键名**：
- `mimic_obs`: 35维模仿观察
- `neck_data`: 颈部控制数据
- `controller_data`: VR控制器数据

---

### 阶段三：运动重定向核心（3-4天）

#### 3.1 GMR (General Motion Retargeting) 模块
**目标**：理解如何将人体动作转换为机器人动作

**关键文件**：
- `GMR/general_motion_retargeting/motion_retarget.py` - 核心重定向类
  - `GeneralMotionRetargeting.__init__()`: 初始化机器人模型和IK配置
  - `GeneralMotionRetargeting.retarget()`: 执行重定向（核心算法）
  
- `GMR/general_motion_retargeting/ik_configs/` - IK配置文件
  - `xrobot_to_g1.json`: PICO VR到G1的映射配置
  - 定义了人体关节到机器人关节的对应关系

**重定向流程**：
1. 接收人体动作数据（SMPL-X格式）
2. 根据IK配置映射人体关节到机器人关节
3. 使用逆运动学求解器计算机器人关节角度
4. 返回机器人配置（root_pos + root_rot + dof_pos）

**关键概念**：
- **IK Match Table**: 人体-机器人关节映射表
- **Scale Table**: 人体尺寸缩放表
- **Task-based IK**: 基于任务的逆运动学求解

#### 3.2 遥操作数据流
**目标**：理解PICO VR数据如何流转

**关键文件**：
- `deploy_real/xrobot_teleop_to_robot_w_hand.py` - 遥操作主程序
  - `XRobotStreamer`: 从PICO VR接收数据
  - `XRobotTeleopToRobot.run()`: 主循环
    - 获取VR数据 → 运动重定向 → 发送到Redis

**数据流**：
```
PICO VR → XRobotStreamer → SMPL-X数据 → GMR.retarget() → 机器人关节角度 → Redis
```

---

### 阶段四：RL训练环境（4-5天）

#### 4.1 训练环境配置
**目标**：理解RL训练如何设置

**关键文件**：
- `legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py` - G1训练配置
  - 观察空间定义
  - 奖励函数配置
  - 环境参数设置

- `legged_gym/legged_gym/envs/base/humanoid_mimic.py` - 模仿任务基类
  - `compute_observations()`: 计算观察
  - `compute_reward()`: 计算奖励
  - `step()`: 环境步进

#### 4.2 观察空间构建
**关键文件**：
- `legged_gym/legged_gym/envs/base/humanoid_mimic.py`
  - 从运动库中提取目标状态
  - 构建模仿观察（mimic_obs）
  - 构建本体感受观察（proprio_obs）
  - 历史状态缓存

#### 4.3 奖励函数设计
**目标**：理解如何设计奖励函数让机器人模仿人类动作

**关键概念**：
- **姿态奖励**: 关节位置误差
- **速度奖励**: 关节速度误差
- **根状态奖励**: 根位置和旋转误差
- **接触奖励**: 足部接触状态匹配

---

### 阶段五：策略部署（2-3天）

#### 5.1 ONNX模型加载
**关键文件**：
- `deploy_real/server_low_level_g1_real.py` (75-89行)
  - `load_onnx_policy()`: 加载ONNX模型
  - `OnnxPolicyWrapper`: ONNX推理包装器

#### 5.2 实时控制循环
**关键文件**：
- `deploy_real/server_low_level_g1_real.py` (200-300行)
  - 主控制循环
  - 观察构建
  - 策略推理
  - 动作执行

**控制频率**：
- 策略频率：50Hz (可配置)
- 仿真频率：1000Hz (1ms步长)
- 降采样：每20步执行一次策略

#### 5.3 动作平滑
**关键文件**：
- `deploy_real/server_low_level_g1_real.py` (45-72行)
  - `EMASmoother`: 指数移动平均平滑器
  - 用于平滑动作输出，减少抖动

---

### 阶段六：高级功能（3-4天）

#### 6.1 手部控制
**关键文件**：
- `deploy_real/robot_control/dex_hand_wrapper.py` - 灵巧手控制
- `deploy_real/xrobot_teleop_to_robot_w_hand.py` - 手部遥操作

#### 6.2 颈部控制
**关键文件**：
- `GMR/general_motion_retargeting/neck_retarget.py` - 颈部重定向
- 根据头部姿态控制机器人颈部

#### 6.3 数据记录
**关键文件**：
- `deploy_real/server_data_record.py` - 数据记录服务器
- `deploy_real/data_utils/episode_writer.py` - 数据写入器

---

## 📖 推荐阅读顺序（详细版）

### 第一周：系统概览和基础

**Day 1-2: 入口和脚本**
1. 阅读 `README.md` 了解项目整体
2. 阅读所有 `.sh` 脚本，理解启动流程
3. 阅读 `doc/TELEOP.md` 了解遥操作流程

**Day 3-4: 低层控制器**
1. `deploy_real/server_low_level_g1_sim.py` - 仿真控制器（完整阅读）
2. `deploy_real/server_low_level_g1_real.py` - 真实控制器（完整阅读）
3. 理解观察空间构建（131-200行）
4. 理解控制循环（200-350行）

**Day 5-7: 数据流和通信**
1. `deploy_real/data_utils/params.py` - 数据定义
2. `deploy_real/server_motion_lib.py` - 运动服务器
3. 理解Redis通信协议
4. 理解观察空间格式

### 第二周：运动重定向

**Day 8-10: GMR核心**
1. `GMR/general_motion_retargeting/motion_retarget.py` - 完整阅读
2. `GMR/general_motion_retargeting/ik_configs/xrobot_to_g1.json` - IK配置
3. `GMR/general_motion_retargeting/kinematics_model.py` - 运动学模型
4. 理解IK求解过程

**Day 11-12: 遥操作**
1. `deploy_real/xrobot_teleop_to_robot_w_hand.py` - 完整阅读
2. `GMR/general_motion_retargeting/xrobot_utils.py` - VR工具
3. 理解状态机（idle/teleop/pause/exit）
4. 理解手部控制

**Day 13-14: 集成测试**
1. 运行 `sim2sim.sh` 测试仿真
2. 运行 `teleop.sh` 测试遥操作
3. 理解完整数据流

### 第三周：RL训练

**Day 15-17: 训练环境**
1. `legged_gym/legged_gym/envs/base/humanoid_mimic.py` - 模仿任务
2. `legged_gym/legged_gym/envs/g1/g1_mimic_future.py` - G1任务
3. `legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py` - 配置
4. 理解观察空间构建
5. 理解奖励函数

**Day 18-19: 训练脚本**
1. `legged_gym/legged_gym/scripts/train.py` - 训练主脚本
2. `rsl_rl/rsl_rl/` - RL算法实现
3. 理解训练流程

**Day 20-21: 模型导出**
1. `legged_gym/legged_gym/scripts/save_onnx.py` - ONNX导出
2. `to_onnx.sh` - 导出脚本
3. 理解模型转换

---

## 🔍 关键代码位置速查

### 观察空间
- **定义**: `deploy_real/data_utils/params.py`
- **构建（仿真）**: `deploy_real/server_low_level_g1_sim.py:150-200`
- **构建（真实）**: `deploy_real/server_low_level_g1_real.py:200-280`
- **训练环境**: `legged_gym/legged_gym/envs/base/humanoid_mimic.py:compute_observations()`

### 动作执行
- **仿真**: `deploy_real/server_low_level_g1_sim.py:250-300`
- **真实**: `deploy_real/server_low_level_g1_real.py:280-350`
- **PD控制**: `robot_control/g1_wrapper.py`

### 运动重定向
- **核心类**: `GMR/general_motion_retargeting/motion_retarget.py:GeneralMotionRetargeting`
- **重定向函数**: `motion_retarget.py:retarget()`
- **IK配置**: `GMR/general_motion_retargeting/ik_configs/`

### Redis通信
- **发送（高层）**: `deploy_real/server_motion_lib.py:send_to_redis()`
- **发送（遥操作）**: `deploy_real/xrobot_teleop_to_robot_w_hand.py:send_to_redis()`
- **接收（低层）**: `deploy_real/server_low_level_g1_real.py:get_mimic_obs_from_redis()`

---

## 💡 学习建议

1. **先运行后理解**：先让系统跑起来，再深入代码
2. **画流程图**：理解数据流时，画出流程图帮助理解
3. **打印调试**：在关键位置添加print，观察数据变化
4. **分模块学习**：不要试图一次性理解所有代码
5. **参考论文**：结合论文理解设计思路

---

## 🐛 常见问题

### Q: 观察空间为什么是1402维？
A: 35 (mimic_obs) + 92 (proprio) = 127 (单帧)
   127 * 11 (当前帧 + 10帧历史) = 1397
   1397 + 35 (当前mimic_obs) = 1432
   实际可能是1402，需要查看具体实现

### Q: 为什么需要历史状态？
A: RL策略需要时序信息来预测未来动作，历史状态帮助策略理解运动趋势

### Q: Mimic Obs和Proprio Obs的区别？
A: 
- **Mimic Obs**: 目标状态（应该达到的状态），来自运动重定向
- **Proprio Obs**: 当前状态（实际状态），来自机器人传感器

### Q: 如何添加新的机器人支持？
A: 
1. 在 `GMR/general_motion_retargeting/params.py` 添加机器人定义
2. 创建IK配置文件 `ik_configs/xxx_to_robot.json`
3. 在 `deploy_real/data_utils/params.py` 添加默认观察值

---

## 📚 相关资源

- **论文**: `2511.02832v1.pdf`
- **文档**: `doc/` 目录
- **GMR文档**: `GMR/README.md`
- **项目网站**: https://yanjieze.com/TWIST2

---

**祝学习顺利！如有问题，可以查看代码注释或参考论文。**

