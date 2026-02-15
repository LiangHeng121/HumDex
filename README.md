# G1 Sim2Real + Wuji 手 + 数据采集 使用指南（正式版）

## 1. 目的与组件

本流程用于在 **G1 真机**上同时完成：
- **身体控制**：本机运行 `sim2real.sh`（低层控制器）
- **Wuji 灵巧手控制**：本机通过 SSH 在 g1 上运行 `wuji_hand_redis_g1.sh`
- **Teleop 数据发送**：本机运行 `teleop.sh`（向 Redis 发布 `action_*` 与 `hand_tracking_*`，并支持键盘开关 `k`）
- **数据采集**：本机运行数据录制脚本，将图像 + state/action + Wuji 字段落盘（推荐键盘录制 `r/q`）

> “第一视角（PICO 显示）”为独立模块，本指南先预留占位（见 §5），后续再补。

---

## 2. 前置条件（网络/Redis/代码）

### 2.1 Redis 必须统一

所有进程必须对齐到同一个 Redis 拓扑，否则会出现“身体能动但手不动 / 录不到数据 / 录到旧数据”等问题。

本仓库默认推荐拓扑：
- **Redis 运行在本机（电脑）**：`redis_ip=localhost`
- **本机进程（teleop / sim2real / recorder）**：连接 `localhost:6379`
- **g1 上的 Wuji 手进程**：连接到 **本机在机器人网络下可达的 IP**（例如 `192.168.123.222` / `172.20.10.5` 等）

### 2.2 g1 侧代码路径

`wuji_hand_redis_g1.sh` / `realsense_zmq_pub_g1.sh` 会在 g1 上寻找 `TWIST2` 仓库目录（需要包含 `deploy_real/`）。

---

## 3. 操作步骤（推荐启动顺序）

### 3.1 打开 g1 并进入模式

1. 打开 g1 电源
2. 进入 **阻尼模式**
3. 进入 **调试模式**

### 3.1.1 （推荐）每次录制前先清理 Redis 旧缓存（本机）

为避免录制读到 Redis 里残留的旧值（例如上一次运行留下的 `state_* / action_* / hand_tracking_* / wuji_*`），建议**每次开始录制前**先执行一次精确清理：

```bash
bash clear_twist2_redis_keys.sh --apply
```

说明：
- 这是“精确删除相关 key”，不会 `flushdb` 清空整个 Redis。
- 如果你的 Redis 不在本机，把 `--redis_ip` 改成实际 Redis 地址。

### 3.2 启动身体控制（本机）

在本机运行：

```bash
bash sim2real.sh
```

按遥控器提示操作：
- **START**：切换到默认站立位
- **A**：进入控制循环（开始执行策略）
- **Select**：结束控制并退出（推荐的“结束控制 g1”的方式）

### 3.3 启动 Wuji 手控制（本机 → SSH 到 g1）

在本机运行（示例）：

```bash
bash wuji_hand_redis_g1.sh --hand_side right --redis_ip <本机IP>
```

说明：
- `--redis_ip` 必须是 **g1 能访问到的本机 IP**（不要填 `localhost`，否则 g1 会连到它自己的 localhost）。
- 左手将 `--hand_side right` 改为 `left`。

#### 3.3.1 同时连接两只 Wuji 手（推荐 + 多设备筛选）

当系统中连接了多台 Wuji 灵巧手时，必须通过序列号筛选设备（避免 `wujihandpy.Hand()` 无参构造报错）。

示例（同时启动左右手）：

```bash
bash wuji_hand_redis.sh \
  --hand_side both \
  --left_serial 3473384E3433 \
  --right_serial 3478385B3433
```

> 如只启动单手，也可以使用 `--serial_number <sn>` 指定设备。

### 3.4 启动 Teleop（本机）

在本机运行：

```bash
bash teleop.sh
```

说明：
- `teleop.sh` 当前运行的是 `deploy_real/xrobot_teleop_to_robot_w_hand_keyboard.py`
- Teleop 键盘控制（与录制按键不冲突）：
  - **k**：切换是否向 Redis 发送数据（`send_enabled` True/False）
    - **全身**：关闭发送时会持续发布“安全默认动作”（`DEFAULT_MIMIC_OBS[:35]` + 手7D全0 + 脖子[0,0]），避免低层继续执行旧指令。
    - **Wuji**：同时写入 `wuji_hand_mode_{left/right}_unitree_g1_with_hands=default`，由 Wuji 控制器在 g1 上使用自身 `zero_pose` 回零位（无需 hand_tracking）。
  - **p**：切换“保持当前位置 / 冻结”（`hold_position_enabled` True/False）
    - **全身**：开启后持续发布**最后一次动作**（body/hand/neck），实现“保持当前位置/姿态”。
    - **Wuji**：同时写入 `wuji_hand_mode_{left/right}_unitree_g1_with_hands=hold`，由 Wuji 控制器在 g1 上重复下发 `last_qpos` 保持原位（无需 hand_tracking）。
    - 若 `k` 关闭发送，则会自动退出保持模式（因为此时全身进入“安全默认动作”，Wuji 进入 `default` 回零位）。

#### 3.4.1 如何修改 **k** 的 default pose（推荐流程）

> 你要改的其实是两套默认：
> - **全身 default（body/neck/Unitree 7D 手）**：由 teleop 在 `k` 分支写入 `action_*` 决定。
> - **Wuji default（灵巧手 20D）**：由 g1 上 Wuji 控制器的 `zero_pose` 决定（teleop 只写 `wuji_hand_mode=default`）。

**A) 修改全身 default（常用）**
1. 先把机器人/仿真调到你想要的“默认站姿/抬手/脖子角度/夹爪姿态”。
2. 运行抓取脚本，打印当前 `action_*`（推荐用 action，因为这就是低层执行的目标）：

```bash
python deploy_real/capture_current_pose_from_redis.py --redis_ip localhost --source action
```

> 如果你不想依赖 Redis（例如你想“直接用当前真机姿态当 default”），也可以在能直连 G1 的环境运行：
>
> ```bash
> python deploy_real/capture_current_pose_from_redis.py --source g1 --net eno1
> ```
>
> 说明：`--source g1` 会用 G1 的 IMU（roll/pitch/yaw_ang_vel）+ 关节角（29D）拼出 35D mimic_obs；其中高度 `root_pos_z` 默认用 `0.79`（可用 `--g1_height` 覆盖）。

3. 修改 `deploy_real/xrobot_teleop_to_robot_w_hand_keyboard.py`：
   - 找到 `send_to_redis()` 里的 `if not send_enabled:` 分支
   - 用脚本打印出来的 `BODY_35 / HAND_LEFT_7 / HAND_RIGHT_7 / NECK_2` 替换默认值

**B) 修改 Wuji default（可选）**
- 修改 g1 上 Wuji 控制脚本的 `zero_pose`（目前是“全 0”）：
  - 单手：`deploy_real/server_wuji_hand_redis.py`
  - 双手单进程：`deploy_real/server_wuji_hands_redis_dual.py`
- 说明：按 `k` 时 teleop 会写 `wuji_hand_mode=default`，Wuji 控制器会下发 `zero_pose`，所以只要改 `zero_pose` 就能改变“k 的默认手姿态”。

### 3.4.2 （可选）离线动作回放：BVH Replay（只控制全身）

> 场景：你手上有一个“全身动作 BVH 文件”，想直接回放到真机（通过 Redis 写 `action_body_*`）。
> 目前版本 **只控制全身**，不控制：
> - Wuji 灵巧手（会被强制置为 `default` 回零位）
> - Unitree 夹爪手（7D 全 0）
> - 脖子（`[0,0]`）

**启动顺序**
1. `bash sim2real.sh`（低层必须在跑，否则写 Redis 没有效果）
2. 不要同时运行 `teleop.sh`（避免两个进程同时写 `action_*` 互相覆盖）
3. 运行 replay：

```bash
bash replay_bvh.sh \
  --bvh_file <你的.bvh路径> \
  --format nokov \
  --redis_ip localhost \
  --fps 30
```

常用参数：
- `--format {lafan1,nokov}`：BVH 格式（不确定时先试 `lafan1`，不对再换 `nokov`）
- `--loop`：循环播放
- `--start / --end`：截取帧区间
- `--offset_to_ground`：把动作贴地（常用）

### 3.4.3 （可选）离线动作回放：BVH Replay（只控制 Wuji 手）

> 场景：你想用 BVH 里的手指/手腕动作，直接回放到 Wuji 灵巧手（通过 Redis 写 `hand_tracking_*`）。
> 该回放脚本 **只控制 Wuji**，不会写全身 `action_*`。

**启动顺序**
1. 在 g1 上启动 Wuji 控制器（单手或双手都可），并确保它连接到本机 Redis。
2. 本机运行 Wuji replay（会写 `wuji_hand_mode_* = follow` + `hand_tracking_*`）：

```bash
bash replay_bvh_wuji.sh \
  --bvh_file <你的.bvh路径> \
  --format nokov \
  --redis_ip localhost \
  --fps 30 \
  --hands both
```

常用参数：
- `--hands {left,right,both}`：只回放左手/右手/双手
- `--loop`：循环播放
- `--start / --end`：截取帧区间

### 3.4.4 实时遥操：XDMocap → 全身（不含手/脖子/Wuji）

> 场景：用 XDMocap 的 UDP 实时数据驱动 G1 全身（通过 GMR 重定向后写入 Redis 的 `action_body_*`）。
> 当前版本 **只控制全身**，不控制：
> - Unitree 夹爪手（7D 全 0）
> - 脖子（`[0,0]`）
> - Wuji（强制 `wuji_hand_mode=default`，并禁用 `hand_tracking_*`）

**启动顺序**
1. `bash sim2real.sh`（低层必须在跑）
2. 确认 XDMocap 广播 IP/端口（例：`192.168.31.134:7000`）
3. 本机启动遥操：

```bash
bash xdmocap_teleop_body.sh --mocap_ip <广播IP> --mocap_port 7000 --world_space geo --fps 60 --offset_to_ground
```

参数说明：
- `--world_space {geo,unity,ue4}`：与 SDK 一致
- `--format {lafan1,nokov}`：选择匹配的 GMR BVH 配置风格（默认 lafan1）
- `--apply_bvh_rotation`：对齐 BVH/GMR 的坐标系（如果你发现方向不对/左右反了，可以开/关对比）

### 3.5 第一视角（用于 PICO 显示）

> 说明（更新）：第一视角与录制现在**共用同一条 RealSense 采集链路**，避免“相机占用冲突”。
>
> - **g1 上只启动一个相机进程**：`XRoboToolkit-Orin-Video-Sender/OrinVideoSender`
> - 它会同时输出：
>   - **第一视角到 VR/PICO（TCP）**
>   - **录制用图像到本机（ZMQ RAW，默认端口 5556）**
>
> PICO 显示侧的具体配置/投屏细节仍先占位（见 §5）。

### 3.6 启动第一视角 + 录制用图像发布（g1 上 OrinVideoSender，推荐）

方式 A：本机 SSH 一键启动（推荐）：

```bash
bash orin_realsense_sender_g1.sh
```

说明：
- `orin_realsense_sender_g1.sh` 默认会在 g1 上执行 `~/XRoboToolkit-Orin-Video-Sender/realsense.sh`
- 当前 `realsense.sh` 已写死（单行）：
  - `--listen 0.0.0.0:13579`
  - `--zmq-raw tcp://*:5556`
- 因此本机会从 `tcp://<G1_IP>:5556` 订阅 raw 图像用于录制

方式 B：直接在 g1 上运行（用于现场调试）：

```bash
cd ~/XRoboToolkit-Orin-Video-Sender
./realsense.sh
```

> 如果提示 RealSense busy，优先确认是否有残留 `videohub_pc4` / 其他占用 `/dev/video*` 的进程。

### 3.6.1 旧方案（可选）：RealSense → ZMQ JPEG（`realsense_zmq_pub_g1.sh`）

> 仅在你明确需要“JPEG/VisionClient 协议”（端口 5555）时使用。
> 注意：它会与 OrinVideoSender **争用同一个 RealSense**，两者**不能同时运行**。

```bash
bash realsense_zmq_pub_g1.sh --host g1 --remote_dir ~/TWIST2 --port 5555 --width 640 --height 480 --fps 30
```

### 3.7 启动数据录制（本机）

#### 方案 A（推荐，无手柄）：键盘录制脚本

```bash
bash data_record_keyboard.sh
```

按键：
- **r**：开始/停止录制（停止会触发保存）
- **q**：退出录制程序

#### 方案 B（兼容旧流程，依赖手柄）：`data_record.sh`

```bash
bash data_record.sh
```

注意：
- `data_record.sh` 对应的 `server_data_record.py` 依赖 Redis 中的 `controller_data`（手柄按键触发录制），如果你“不用手柄”，建议使用方案 A。

---

## 4. 数据与字段说明（Redis Key / 形状 / 落盘结构）

### 4.1 Redis Key（核心）

#### 身体与脖子（由 teleop 写 action、低层写 state）
- **action（teleop → Redis）**
  - `action_body_unitree_g1_with_hands`：35
  - `action_hand_left_unitree_g1_with_hands`：7
  - `action_hand_right_unitree_g1_with_hands`：7
  - `action_neck_unitree_g1_with_hands`：2
  - `t_action`：毫秒时间戳（int）
- **state（低层 → Redis）**
  - `state_body_unitree_g1_with_hands`：34（`ang_vel(3) + roll/pitch(2) + dof_pos(29)`）
  - `state_hand_left_unitree_g1_with_hands`：通常 7（若低层启用手）
  - `state_hand_right_unitree_g1_with_hands`：通常 7（若低层启用手）
  - `t_state`：毫秒时间戳（int；是否存在取决于低层实现）

#### Wuji 手输入（teleop → Redis）
- `hand_tracking_left_unitree_g1_with_hands`：dict（含 `is_active`、`timestamp` + 26 个关节字段）
- `hand_tracking_right_unitree_g1_with_hands`：同上
- `wuji_hand_mode_left_unitree_g1_with_hands`：字符串（`follow` / `hold` / `default`）
- `wuji_hand_mode_right_unitree_g1_with_hands`：字符串（`follow` / `hold` / `default`）

#### Wuji 手 action/state（g1 上的 Wuji 控制器写回 Redis）
由 `deploy_real/server_wuji_hand_redis.py` 写回，用于录制/排障：
- **action（retarget 后目标）**
  - `action_wuji_qpos_target_left_unitree_g1_with_hands`：20（flatten list）
  - `action_wuji_qpos_target_right_unitree_g1_with_hands`：20
  - `t_action_wuji_hand_left_unitree_g1_with_hands`：ms
  - `t_action_wuji_hand_right_unitree_g1_with_hands`：ms
- **state（硬件反馈）**
  - `state_wuji_hand_left_unitree_g1_with_hands`：20（flatten list）
  - `state_wuji_hand_right_unitree_g1_with_hands`：20
  - `t_state_wuji_hand_left_unitree_g1_with_hands`：ms
  - `t_state_wuji_hand_right_unitree_g1_with_hands`：ms

### 4.2 图像（ZMQ）

当前有两种图像发布格式（根据你用哪套相机发布端）：

#### A) JPEG（`VisionClient` 协议，`realsense_zmq_pub_g1.sh`）
- **消息格式**：`[int32 width][int32 height][int32 jpeg_len][jpeg_bytes]`
- **端口（默认）**：5555
- **录制端**：`--vision_backend zmq`

#### B) RAW（OrinVideoSender `--zmq-raw`，推荐）
- **消息格式**：`[int32 width][int32 height][int32 channels][raw_bytes]`
  - `channels` 通常为 3(BGR) 或 4(BGRA)
- **端口（默认）**：5556（与 `XRoboToolkit-Orin-Video-Sender/realsense.sh` 对齐）
- **录制端**：`--vision_backend orin_zmq_raw`

录制端（本机）解码后记录：
- `rgb`：形状 `(H, W, 3)`，用于 `cv2.imwrite` 落盘，**按 OpenCV 习惯为 BGR**

> 具体分辨率取决于 g1 发布端 `--width/--height`，以及录制脚本配置的 `img_shape`。

### 4.3 落盘目录结构（EpisodeWriter）

录制目录：
- `<data_folder>/<task_name>/episode_0000/`
  - `rgb/000000.jpg`
  - `rgb/000001.jpg`
  - `data.json`

`data.json` 结构：
- `info`：元信息（版本、日期、图像 fps 等）
- `text`：文本描述（goal/desc/steps）
- `data`：列表，每一项是一帧样本，包含：
  - `idx`
  - `rgb`（相对路径）
  - `t_img` / `t_record_ms`
  - `state_*` / `action_*` / `hand_tracking_*` / `wuji_*`（若当时可读到）


