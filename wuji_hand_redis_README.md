# Wuji Hand Controller via Redis

从 Redis 读取 `teleop.sh` 发送的手部控制数据，实时控制 Wuji 灵巧手。

## 功能说明

- 从 Redis 读取 `teleop.sh` 发送的手部控制数据（7D Unitree G1 格式）
- 将 7D 数据转换为 Wuji 手的 20D (5×4) 关节数据
- 实时控制 Wuji 灵巧手，支持平滑移动
- 支持左手和右手控制

## 使用方法

### 1. 前置条件

- 已安装 `wujihandpy` 库
- 已连接 Wuji 灵巧手硬件
- Redis 服务正在运行
- `teleop.sh` 正在运行并发送手部数据到 Redis

### 2. 运行脚本

```bash
# 使用默认配置（左手，localhost Redis，50Hz）
./wuji_hand_redis.sh

# 或者直接运行 Python 脚本
cd deploy_real
python server_wuji_hand_redis.py --hand_side left --redis_ip localhost
```

### 3. 参数说明

```bash
python server_wuji_hand_redis.py \
    --hand_side left          # 控制左手或右手 (left/right)
    --redis_ip localhost      # Redis 服务器 IP
    --target_fps 50           # 目标控制频率 (Hz)
    --no_smooth               # 禁用平滑移动（直接设置目标位置）
    --smooth_steps 5          # 平滑移动步数（默认: 5）
```

## 工作流程

1. **初始化**
   - 连接 Redis 服务器
   - 初始化 Wuji 手硬件
   - 获取零位位置

2. **主循环**
   - 从 Redis 读取手部数据（键名: `action_hand_left_unitree_g1_with_hands` 或 `action_hand_right_unitree_g1_with_hands`）
   - 将 7D Unitree 格式转换为 20D Wuji 格式
   - 平滑移动到目标位置
   - 控制频率（默认 50Hz）

3. **退出**
   - 平滑回到零位
   - 关闭控制器
   - 失能电机

## 数据格式转换

### Unitree G1 格式 (7D)
```
[thumb_base, thumb, index, middle, ring, pinky, wrist]
```

### Wuji 手格式 (20D = 5×4)
```
5 个手指 × 4 个关节 = 20 个关节
形状: (5, 4)
```

### 映射规则

当前实现使用启发式映射：
- 拇指 (finger 0): 使用 `thumb_base` 和 `thumb`
- 食指 (finger 1): 使用 `index`
- 中指 (finger 2): 使用 `middle`
- 无名指 (finger 3): 使用 `ring`
- 小指 (finger 4): 使用 `pinky`

每个手指的 4 个关节按比例缩放。

**注意**: 映射规则可能需要根据实际情况调整。可以在 `unitree_7d_to_wuji_20d()` 函数中修改。

## 与 teleop.sh 的配合

1. **启动 teleop.sh**（发送手部数据到 Redis）
   ```bash
   ./teleop.sh
   ```

2. **启动 wuji_hand_redis.sh**（从 Redis 读取并控制 Wuji 手）
   ```bash
   ./wuji_hand_redis.sh
   ```

3. **同时运行**（两个终端）
   - 终端 1: `./teleop.sh` - PICO VR 遥操作
   - 终端 2: `./wuji_hand_redis.sh` - Wuji 手控制

## 故障排除

### 1. Redis 连接失败
```
❌ Redis 连接失败: Connection refused
```
**解决**: 确保 Redis 服务正在运行
```bash
redis-cli ping  # 应该返回 PONG
```

### 2. Wuji 手初始化失败
```
❌ 错误: 未安装 wujihandpy
```
**解决**: 安装 wujihandpy
```bash
pip install wujihandpy
```

### 3. 没有手部数据
如果 Redis 中没有手部数据，Wuji 手会保持上次位置。确保 `teleop.sh` 正在运行。

### 4. 映射不准确
如果手部动作不准确，可能需要调整 `unitree_7d_to_wuji_20d()` 函数中的映射规则。

## 参考

- `teleop_realtime.py`: Wuji 手实时控制示例
- `server_low_level_g1_real.py`: Redis 通信示例
- `xrobot_teleop_to_robot_w_hand.py`: 手部数据发送到 Redis 的代码

