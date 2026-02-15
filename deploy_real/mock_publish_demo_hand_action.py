#!/usr/bin/env python3
"""
Mock Hand Tracking Publisher

从 demo JSON 文件读取手部追踪数据，按顺序发布到 Redis，
用于测试 server_wuji_hand_redis.py。

Usage:
    python mock_hand_tracking_publisher.py --json_file /path/to/demo.json --hand_side left
    python mock_hand_tracking_publisher.py --json_file /path/to/demo.json --hand_side right --fps 10
"""

import argparse
import json
import time
import sys

try:
    import redis
except ImportError:
    print("Error: redis package not installed. Run: pip install redis")
    sys.exit(1)


def now_ms() -> int:
    """当前时间戳（毫秒）"""
    return int(time.time() * 1000)


class MockHandTrackingPublisher:
    """从 demo JSON 读取手部追踪数据并发布到 Redis"""

    def __init__(self, json_file: str, hand_side: str, redis_ip: str = "localhost", fps: float = 30.0):
        """
        Args:
            json_file: demo JSON 文件路径
            hand_side: "left" 或 "right"
            redis_ip: Redis 服务器 IP
            fps: 发布频率 (Hz)
        """
        self.json_file = json_file
        self.hand_side = hand_side.lower()
        assert self.hand_side in ["left", "right"], "hand_side must be 'left' or 'right'"

        self.fps = fps
        self.dt = 1.0 / fps

        # Redis key 名（与 server_wuji_hand_redis.py 一致）
        self.robot_key = "unitree_g1_with_hands"
        self.redis_key_hand_tracking = f"hand_tracking_{self.hand_side}_{self.robot_key}"
        self.redis_key_wuji_mode = f"wuji_hand_mode_{self.hand_side}_{self.robot_key}"

        # 连接 Redis
        print(f"Connecting to Redis: {redis_ip}:6379")
        try:
            self.redis_client = redis.Redis(host=redis_ip, port=6379, decode_responses=False)
            self.redis_client.ping()
            print("Redis connected successfully")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            raise

        # 加载 demo 数据
        print(f"Loading demo data from: {json_file}")
        self.frames = self._load_demo_data()
        print(f"Loaded {len(self.frames)} frames")

    def _load_demo_data(self):
        """加载 demo JSON 并提取手部追踪数据"""
        with open(self.json_file, 'r') as f:
            demo = json.load(f)

        frames = []
        data_list = demo.get("data", [])

        # 选择对应手的 key
        tracking_key = f"hand_tracking_{self.hand_side}"

        for item in data_list:
            hand_data = item.get(tracking_key)
            if hand_data is not None and isinstance(hand_data, dict):
                # 检查是否有有效数据（至少有 Wrist）
                wrist_key = f"{self.hand_side.capitalize()}HandWrist"
                if wrist_key in hand_data or "LeftHandWrist" in hand_data or "RightHandWrist" in hand_data:
                    frames.append(hand_data)

        if len(frames) == 0:
            print(f"Warning: No valid hand tracking data found for '{self.hand_side}' hand")
            print(f"  Checked key: '{tracking_key}'")
            # 打印第一个 item 的 keys 以便调试
            if data_list:
                print(f"  Available keys in first frame: {list(data_list[0].keys())}")

        return frames

    def publish_frame(self, frame_data: dict):
        """发布单帧手部追踪数据到 Redis"""
        # 更新时间戳（必须是新鲜的，否则 server 会认为数据过期）
        publish_data = frame_data.copy()
        publish_data["timestamp"] = now_ms()
        publish_data["is_active"] = True

        # 发布到 Redis
        self.redis_client.set(self.redis_key_hand_tracking, json.dumps(publish_data))

        # 设置模式为 follow（让 Wuji server 跟随数据）
        self.redis_client.set(self.redis_key_wuji_mode, "follow")

    def run(self, loop: bool = True):
        """
        主循环：按顺序发布帧数据（ping-pong 模式）

        播放顺序: 0 -> 1 -> ... -> N-1 -> N-2 -> ... -> 1 -> 0 -> 1 -> ...
        这样避免头尾之间动作跳变过大。

        Args:
            loop: 是否循环播放
        """
        if len(self.frames) == 0:
            print("No frames to publish. Exiting.")
            return

        print(f"\nStarting to publish {self.hand_side} hand tracking data")
        print(f"  Redis key: {self.redis_key_hand_tracking}")
        print(f"  FPS: {self.fps} Hz (interval: {self.dt*1000:.1f} ms)")
        print(f"  Loop: {loop}")
        print(f"  Mode: ping-pong (forward -> backward -> forward ...)")
        print("\nPress Ctrl+C to stop\n")

        frame_idx = 0
        direction = 1  # 1 = forward, -1 = backward
        total_published = 0
        loop_count = 0

        try:
            while True:
                loop_start = time.time()

                # 获取当前帧
                frame_data = self.frames[frame_idx]

                # 发布
                self.publish_frame(frame_data)
                total_published += 1

                # 打印进度（每 10 帧打印一次）
                if total_published % 10 == 0:
                    dir_str = ">>>" if direction == 1 else "<<<"
                    print(f"{dir_str} Frame {frame_idx + 1}/{len(self.frames)} (total: {total_published}, loops: {loop_count})", end='\r')

                # 下一帧 (ping-pong)
                frame_idx += direction

                # 到达尾部，反向
                if frame_idx >= len(self.frames):
                    frame_idx = len(self.frames) - 2  # 回退一帧避免重复最后一帧
                    direction = -1
                    if frame_idx < 0:
                        frame_idx = 0
                    print(f"\n<<< Reached end, reversing direction...")

                # 到达头部，反向
                elif frame_idx < 0:
                    frame_idx = 1  # 前进一帧避免重复第一帧
                    direction = 1
                    loop_count += 1
                    if frame_idx >= len(self.frames):
                        frame_idx = 0
                    if loop:
                        print(f"\n>>> Reached start, reversing direction (loop {loop_count})...")
                    else:
                        print(f"\nPing-pong completed. Exiting.")
                        break

                # 控制频率
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n\nStopped. Total frames published: {total_published}")
        finally:
            # 发送 is_active=False 表示停止
            stop_data = {"is_active": False, "timestamp": now_ms()}
            self.redis_client.set(self.redis_key_hand_tracking, json.dumps(stop_data))
            print("Published is_active=False to signal stop")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Mock hand tracking publisher for testing Wuji hand server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Publish left hand data at 30 FPS
    python mock_hand_tracking_publisher.py --json_file demo.json --hand_side left

    # Publish right hand data at 10 FPS (slower)
    python mock_hand_tracking_publisher.py --json_file demo.json --hand_side right --fps 10

    # Publish once without looping
    python mock_hand_tracking_publisher.py --json_file demo.json --hand_side left --no_loop
        """
    )

    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to demo JSON file containing hand tracking data"
    )

    parser.add_argument(
        "--hand_side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Which hand to publish (default: left)"
    )

    parser.add_argument(
        "--redis_ip",
        type=str,
        default="localhost",
        help="Redis server IP (default: localhost)"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Publish frequency in Hz (default: 30.0)"
    )

    parser.add_argument(
        "--no_loop",
        action="store_true",
        help="Don't loop - stop after publishing all frames once"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 60)
    print("Mock Hand Tracking Publisher")
    print("=" * 60)
    print(f"JSON file: {args.json_file}")
    print(f"Hand side: {args.hand_side}")
    print(f"Redis IP: {args.redis_ip}")
    print(f"FPS: {args.fps}")
    print(f"Loop: {not args.no_loop}")
    print("=" * 60)

    publisher = MockHandTrackingPublisher(
        json_file=args.json_file,
        hand_side=args.hand_side,
        redis_ip=args.redis_ip,
        fps=args.fps
    )

    publisher.run(loop=not args.no_loop)


if __name__ == "__main__":
    main()