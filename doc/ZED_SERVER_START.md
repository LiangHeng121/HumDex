# ZED 相机服务器启动说明

## 概述

ZED 相机服务器运行在机器人上，负责捕获 ZED 立体相机图像并通过 ZeroMQ 发布到网络。

## 脚本位置

`docker_zed.sh` 脚本位于**机器人上**的以下路径：
```
~/g1-onboard/docker_zed.sh
```

**注意**：这个脚本不在本地的 TWIST2 代码库中，而是在机器人（G1）的 `~/g1-onboard/` 目录下。

## 启动方式

### 方式 1：通过 GUI 启动（推荐）

1. 启动 GUI：
   ```bash
   bash gui.sh
   ```

2. 在 GUI 界面中：
   - 找到 "G1 ZED Teleop" 面板
   - 点击 "START" 按钮
   - 或者点击 "🚀 Start Neck & ZED Teleop" 按钮（会同时启动 Neck 和 ZED 服务器）

3. 查看状态：
   - 面板状态会显示 "RUNNING"（绿色）
   - 可以在终端输出中查看日志

### 方式 2：通过 SSH 手动启动

1. **确保 SSH 配置正确**：
   - 确保你的 `~/.ssh/config` 中配置了 `g1` 主机：
     ```
     Host g1
         HostName <机器人IP地址>
         User <用户名>
         # 可选：配置密钥认证
         IdentityFile ~/.ssh/id_rsa
     ```
   - 或者直接使用 IP 地址：`ssh user@192.168.123.164`

2. **SSH 连接到机器人**：
   ```bash
   ssh g1
   # 或
   ssh user@192.168.123.164
   ```

3. **在机器人上执行启动脚本**：
   ```bash
   cd ~
   bash ~/g1-onboard/docker_zed.sh
   ```

### 方式 3：从本地直接 SSH 执行

```bash
ssh g1 "cd ~ && bash ~/g1-onboard/docker_zed.sh"
```

或者使用 IP 地址：
```bash
ssh user@192.168.123.164 "cd ~ && bash ~/g1-onboard/docker_zed.sh"
```

## 停止服务器

### 方式 1：通过 GUI 停止

在 GUI 的 "G1 ZED Teleop" 面板中点击 "KILL" 按钮。

### 方式 2：手动停止

在机器人上执行：
```bash
pkill -9 OrinVideoSender
```

或者从本地 SSH 执行：
```bash
ssh g1 "pkill -9 OrinVideoSender"
```

## 验证服务器是否运行

### 1. 检查进程

在机器人上执行：
```bash
ps aux | grep OrinVideoSender
```

如果看到进程在运行，说明服务器已启动。

### 2. 检查端口

在机器人上执行：
```bash
netstat -tuln | grep 5555
```

或者：
```bash
ss -tuln | grep 5555
```

如果看到端口 5555 在监听，说明服务器正在运行。

### 3. 测试连接

在本地机器上，可以使用 `VisionClient` 测试连接：
```python
from deploy_real.data_utils.vision_client import VisionClient
import time

client = VisionClient(
    server_address="192.168.123.164",  # 机器人 IP
    port=5555,
    image_show=True  # 显示图像
)

# 启动接收线程
import threading
thread = threading.Thread(target=client.receive_process, daemon=True)
thread.start()

# 等待一段时间查看是否收到图像
time.sleep(5)
```

## 服务器功能

ZED 服务器执行以下功能：

1. **捕获图像**：
   - 使用 ZED 立体相机捕获 RGB 图像
   - 图像尺寸：每个相机 `640×360`，拼接后为 `1280×360`

2. **图像压缩**：
   - 将图像压缩为 JPEG 格式
   - 减少网络传输带宽

3. **ZeroMQ 发布**：
   - 使用 ZeroMQ PUB socket 在端口 `5555` 上发布图像
   - 数据格式：`[4字节宽度][4字节高度][4字节JPEG长度][JPEG数据]`
   - 监听地址：`0.0.0.0:5555`（所有网络接口）

## 网络要求

1. **机器人网络配置**：
   - 机器人 IP：`192.168.123.164`（默认）
   - 确保机器人可以通过网络访问

2. **本地网络配置**：
   - 本地机器需要能够访问机器人 IP
   - 确保防火墙允许端口 5555 的通信

3. **测试网络连接**：
   ```bash
   ping 192.168.123.164
   ```

## 常见问题

### 1. 无法连接到机器人

**问题**：SSH 连接失败

**解决方案**：
- 检查网络连接：`ping <机器人IP>`
- 检查 SSH 配置：`cat ~/.ssh/config`
- 尝试直接使用 IP：`ssh user@192.168.123.164`

### 2. 脚本不存在

**问题**：`~/g1-onboard/docker_zed.sh` 不存在

**解决方案**：
- 确认脚本路径是否正确
- 检查机器人上是否有 `g1-onboard` 目录
- 可能需要从其他位置复制脚本

### 3. 端口被占用

**问题**：端口 5555 已被占用

**解决方案**：
```bash
# 在机器人上查找占用端口的进程
sudo lsof -i :5555
# 或
sudo netstat -tulpn | grep 5555

# 停止占用端口的进程
pkill -9 OrinVideoSender
```

### 4. 无法接收图像

**问题**：`VisionClient` 无法接收到图像

**解决方案**：
- 检查服务器是否正在运行
- 检查网络连接
- 检查防火墙设置
- 确认机器人 IP 地址正确

### 5. 图像质量差或延迟高

**问题**：图像传输延迟或质量差

**解决方案**：
- 检查网络带宽
- 检查网络延迟：`ping <机器人IP>`
- 可能需要调整 JPEG 压缩质量（在 `docker_zed.sh` 脚本中）

## 相关文件

- **GUI 启动代码**：`gui.py` 第 528-531 行
- **VisionClient 接收代码**：`deploy_real/data_utils/vision_client.py`
- **数据收集脚本**：`deploy_real/server_data_record.py`

## 启动顺序建议

根据 `doc/TELEOP.md`，推荐的启动顺序是：

1. 启动 G1 机器人
2. 启动 Neck 服务器：`bash docker_neck.sh`
3. 重新插拔 ZED MINI 相机以确保连接
4. 启动 ZED 服务器：`bash docker_zed.sh`
5. 在 VR 中开始监听 ZED MINI 相机（应该能看到相机画面）

## 注意事项

1. **ZED 相机连接**：
   - 确保 ZED MINI 相机已正确连接到机器人
   - 如果相机无法识别，尝试重新插拔

2. **Docker 环境**：
   - 脚本名称包含 "docker"，可能需要在 Docker 容器中运行
   - 确保 Docker 环境已正确配置

3. **权限问题**：
   - 某些操作可能需要 sudo 权限
   - 确保用户有执行脚本的权限

4. **资源占用**：
   - ZED 服务器会占用一定的 CPU 和网络资源
   - 确保机器人有足够的资源运行服务器

