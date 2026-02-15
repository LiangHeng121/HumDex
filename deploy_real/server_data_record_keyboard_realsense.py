#!/usr/bin/env python3
"""
Data collection script (keyboard-controlled).

Compared to server_data_record.py:
- No controller_data from Redis; use keyboard to start/stop recording.
- Record BOTH:
  - body state/action from sim2real low-level controller (Redis keys: state_* / action_*)
  - hand tracking dicts used by wuji_hand_redis (Redis keys: hand_tracking_left/right_*)
- Vision source:
  - default: ZMQ JPEG stream via VisionClient (compatible with ZED or any server that publishes the same format)
  - optional: RealSense direct capture via pyrealsense2 (run on the machine that has the RealSense connected)
"""

import argparse
import json
import os
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import cv2  # type: ignore
import redis  # type: ignore
from rich import print  # type: ignore

from data_utils.episode_writer import EpisodeWriter
from data_utils.vision_client import VisionClient
from data_utils.evdev_hotkeys import EvdevHotkeys, EvdevHotkeyConfig


def now_ms() -> int:
    return int(time.time() * 1000)


class ZmqVisionSource:
    """Receive JPEG-compressed RGB frames from a ZMQ PUB server via VisionClient into shared memory."""

    def __init__(
        self,
        server_address: str,
        port: int,
        image_shape: Tuple[int, int, int],
        image_show: bool = False,
        unit_test: bool = False,
    ):
        from multiprocessing import shared_memory

        self.image_shape = image_shape
        shm_bytes = int(np.prod(image_shape) * np.uint8().itemsize)
        self.image_shared_memory = shared_memory.SharedMemory(create=True, size=shm_bytes)
        self.image_array = np.ndarray(image_shape, dtype=np.uint8, buffer=self.image_shared_memory.buf)

        self.client = VisionClient(
            server_address=server_address,
            port=port,
            img_shape=image_shape,
            img_shm_name=self.image_shared_memory.name,
            image_show=image_show,
            depth_show=False,
            unit_test=unit_test,
        )
        self.thread = threading.Thread(target=self.client.receive_process, daemon=True)
        self.thread.start()

    def get_rgb(self) -> np.ndarray:
        return self.image_array.copy()

    def close(self):
        # Best-effort: stop loop and cleanup shm
        try:
            self.client.running = False
        except Exception:
            pass
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.image_shared_memory.unlink()
        except Exception:
            pass
        try:
            self.image_shared_memory.close()
        except Exception:
            pass


class RealSenseVisionSource:
    """Directly capture RealSense color (and optional depth) frames using pyrealsense2."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        enable_depth: bool = False,
    ):
        try:
            import pyrealsense2 as rs  # type: ignore
        except Exception as e:
            raise ImportError(
                "未安装 pyrealsense2。若你在笔记本上录制（相机在 g1 上），建议用 --vision_backend zmq 走网络流；"
                "若你要在 g1 本机直连 RealSense，请先安装 pyrealsense2。"
            ) from e

        self._rs = rs
        self._enable_depth = enable_depth
        self._lock = threading.Lock()
        self._running = True
        self._latest_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self._latest_depth: Optional[np.ndarray] = None

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.profile = self.pipeline.start(config)

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        rs = self._rs
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                color_img = np.asanyarray(color.get_data())  # BGR

                depth_img = None
                if self._enable_depth:
                    depth = frames.get_depth_frame()
                    if depth:
                        depth_img = np.asanyarray(depth.get_data())  # uint16 depth

                with self._lock:
                    self._latest_rgb = color_img
                    self._latest_depth = depth_img
            except Exception:
                time.sleep(0.005)

    def get_rgb(self) -> np.ndarray:
        with self._lock:
            return self._latest_rgb.copy()

    def get_depth(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_depth is None else self._latest_depth.copy()

    def close(self):
        self._running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.pipeline.stop()
        except Exception:
            pass


def safe_json_loads(raw: Optional[bytes]) -> Any:
    if raw is None:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_data_folder = os.path.join(here, "twist2_demonstration")
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")

    parser = argparse.ArgumentParser(description="Record vision + (body/hand) state/action from Redis (keyboard control).")

    # storage
    parser.add_argument("--data_folder", default=default_data_folder, help="数据保存根目录")
    parser.add_argument("--task_name", default=cur_time, help="任务名（子目录）")
    parser.add_argument("--frequency", default=30, type=int, help="录制频率 Hz")

    # redis
    parser.add_argument("--redis_ip", default="localhost", help="Redis IP（注意：必须能读到 sim2real/teleop 写入的数据）")
    parser.add_argument("--redis_port", default=6379, type=int, help="Redis 端口")

    # key namespace
    parser.add_argument("--robot_key", default="unitree_g1_with_hands", help="Redis key 后缀，例如 unitree_g1_with_hands")

    # vision
    parser.add_argument("--vision_backend", choices=["zmq", "realsense"], default="zmq", help="图像来源：zmq(网络流) 或 realsense(直连)")
    parser.add_argument("--vision_ip", default="192.168.123.164", help="ZMQ 图像服务器 IP（vision_backend=zmq 才用）")
    parser.add_argument("--vision_port", default=5555, type=int, help="ZMQ 图像服务器端口（vision_backend=zmq 才用）")

    parser.add_argument("--img_h", default=480, type=int)
    parser.add_argument("--img_w", default=640, type=int)
    parser.add_argument("--img_c", default=3, type=int)

    # realsense
    parser.add_argument("--rs_w", default=640, type=int, help="RealSense 宽（vision_backend=realsense）")
    parser.add_argument("--rs_h", default=480, type=int, help="RealSense 高（vision_backend=realsense）")
    parser.add_argument("--rs_fps", default=30, type=int, help="RealSense FPS（vision_backend=realsense）")
    parser.add_argument("--rs_depth", action="store_true", help="保存 RealSense 深度帧（会写进 data.json，不会保存成图片文件）")

    # output video
    parser.add_argument("--save_episode_video", action="store_true", help="每个 episode 保存一份 mp4 到 task_dir/videos/ 下")

    # ui / keyboard
    parser.add_argument("--no_window", action="store_true", help="不显示窗口（默认按键来自 OpenCV 窗口；无窗口建议用 --keyboard_backend evdev）")
    parser.add_argument(
        "--keyboard_backend",
        choices=["opencv", "evdev"],
        default="opencv",
        help="按键后端：opencv(需要窗口在前台) / evdev(全局热键，不依赖前台，但需要 /dev/input 权限)",
    )
    parser.add_argument("--evdev_device", type=str, default="auto", help="evdev 设备路径，如 /dev/input/event3 或 /dev/input/by-id/...；auto 尝试自动选择")
    parser.add_argument("--evdev_grab", action="store_true", help="evdev grab（可能影响系统其他程序收到按键，谨慎使用）")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Keyboard Data Recorder")
    print("=" * 70)
    print(f"Redis: {args.redis_ip}:{args.redis_port}")
    print(f"robot_key: {args.robot_key}")
    print(f"Vision backend: {args.vision_backend}")
    if args.vision_backend == "zmq":
        print(f"Vision ZMQ: tcp://{args.vision_ip}:{args.vision_port}")
        print(f"Image shape (record): ({args.img_h}, {args.img_w}, {args.img_c})")
    else:
        print(f"RealSense: {args.rs_w}x{args.rs_h}@{args.rs_fps}, depth={args.rs_depth}")
    print(f"Save to: {os.path.join(args.data_folder, args.task_name)}")
    print("Keys: press 'r' to start/stop recording; press 'q' to quit")
    print("=" * 70)

    # Redis connection
    try:
        redis_pool = redis.ConnectionPool(
            host=args.redis_ip,
            port=args.redis_port,
            db=0,
            max_connections=10,
            retry_on_timeout=True,
            socket_timeout=0.2,
            socket_connect_timeout=0.2,
        )
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_pipeline = redis_client.pipeline()
        redis_client.ping()
        print(f"✅ Connected to Redis at {args.redis_ip}:{args.redis_port}, DB=0")
    except Exception as e:
        print(f"❌ Error connecting to Redis: {e}")
        return

    # Vision source
    vision = None
    try:
        if args.vision_backend == "zmq":
            image_shape = (args.img_h, args.img_w, args.img_c)
            vision = ZmqVisionSource(
                server_address=args.vision_ip,
                port=args.vision_port,
                image_shape=image_shape,
                image_show=False,
                unit_test=True,
            )
        else:
            vision = RealSenseVisionSource(
                width=args.rs_w,
                height=args.rs_h,
                fps=args.rs_fps,
                enable_depth=args.rs_depth,
            )
    except Exception as e:
        print(f"❌ Vision init failed: {e}")
        return

    # Recorder
    recording = False
    step_count = 0
    task_dir = os.path.join(args.data_folder, args.task_name)
    recorder = EpisodeWriter(
        task_dir=task_dir,
        frequency=args.frequency,
        image_shape=(args.img_h, args.img_w, args.img_c) if args.vision_backend == "zmq" else (args.rs_h, args.rs_w, 3),
        data_keys=["rgb"],
        save_video=bool(args.save_episode_video),
        video_fps=float(args.frequency),
    )

    control_dt = 1.0 / float(args.frequency)
    running = True

    # Compose redis keys
    suffix = args.robot_key
    redis_keys: List[str] = [
        f"state_body_{suffix}",
        f"state_hand_left_{suffix}",
        f"state_hand_right_{suffix}",
        f"state_neck_{suffix}",
        "t_state",
        f"action_body_{suffix}",
        f"action_hand_left_{suffix}",
        f"action_hand_right_{suffix}",
        f"action_neck_{suffix}",
        "t_action",
        f"hand_tracking_left_{suffix}",
        f"hand_tracking_right_{suffix}",
        # Wuji hand (optional, written by server_wuji_hand_redis.py on g1)
        f"action_wuji_qpos_target_left_{suffix}",
        f"action_wuji_qpos_target_right_{suffix}",
        f"t_action_wuji_hand_left_{suffix}",
        f"t_action_wuji_hand_right_{suffix}",
        f"state_wuji_hand_left_{suffix}",
        f"state_wuji_hand_right_{suffix}",
        f"t_state_wuji_hand_left_{suffix}",
        f"t_state_wuji_hand_right_{suffix}",
    ]
    data_dict_keys: List[str] = [
        "state_body",
        "state_hand_left",
        "state_hand_right",
        "state_neck",
        "t_state",
        "action_body",
        "action_hand_left",
        "action_hand_right",
        "action_neck",
        "t_action",
        "hand_tracking_left",
        "hand_tracking_right",
        "action_wuji_qpos_target_left",
        "action_wuji_qpos_target_right",
        "t_action_wuji_hand_left",
        "t_action_wuji_hand_right",
        "state_wuji_hand_left",
        "state_wuji_hand_right",
        "t_state_wuji_hand_left",
        "t_state_wuji_hand_right",
    ]

    window_name = "TWIST2 Data Recorder (keyboard: r=rec start/stop, q=quit)"
    window_enabled = not bool(args.no_window)
    if window_enabled:
        # 无论 keyboard_backend 是 opencv 还是 evdev，都显示窗口（evdev 模式仅显示，不用窗口读按键）
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # optional global hotkeys via evdev
    evdev_keys: Dict[str, bool] = {"r": False, "q": False}

    def _on_hotkey(ch: str) -> None:
        c = (ch or "")[:1].lower()
        if c in evdev_keys:
            evdev_keys[c] = True

    evdev_listener: Optional[EvdevHotkeys] = None
    if str(args.keyboard_backend).lower() == "evdev":
        cfg = EvdevHotkeyConfig(device=str(args.evdev_device), grab=bool(args.evdev_grab))
        evdev_listener = EvdevHotkeys(cfg, callback=_on_hotkey)
        try:
            evdev_listener.start()
            print("[Keyboard] backend=evdev（全局热键，不依赖前台窗口/终端）")
            print(f"[Keyboard] evdev_device={cfg.device} grab={cfg.grab}")
            print("Keys: press 'r' to start/stop recording; press 'q' to quit")
        except Exception as e:
            print(f"❌ evdev 键盘监听启动失败：{e}")
            print("   你可能需要：pip install evdev，以及对 /dev/input/event* 的读权限（root 或加入 input 组）。")
            # 关键：在这里就退出的话，要把 ZMQ 共享内存也释放掉，避免 resource_tracker 报泄漏
            try:
                recorder.close()
            except Exception:
                pass
            try:
                if vision is not None:
                    vision.close()
            except Exception:
                pass
            return

    try:
        while running:
            t0 = time.time()

            # Grab latest image first (for display & recording)
            rgb = vision.get_rgb() if vision is not None else None
            if rgb is None:
                rgb = np.zeros((args.img_h, args.img_w, 3), dtype=np.uint8)

            # 预览窗口（始终显示；按键仅在 backend=opencv 时生效）
            if window_enabled:
                overlay = rgb.copy()
                status = "REC: ON" if recording else "REC: OFF"
                cv2.putText(overlay, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if recording else (0, 0, 255), 2)
                if str(args.keyboard_backend).lower() == "opencv":
                    cv2.putText(overlay, "keys(opencv focus): r=start/stop, q=quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.putText(overlay, "keys(evdev global): r=start/stop, q=quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(window_name, overlay)
                # IMPORTANT: 即使不使用 opencv 读键，也要 waitKey(1) 保持窗口刷新/响应
                key = cv2.waitKey(1) & 0xFF
                if str(args.keyboard_backend).lower() == "opencv":
                    if key == ord("q"):
                        running = False
                        break
                    if key == ord("r"):
                        recording = not recording
                        if recording:
                            if recorder.create_episode():
                                step_count = 0
                                print("✅ episode recording started")
                            else:
                                recording = False
                        else:
                            recorder.save_episode()
                            print("✅ episode saving triggered")

            if str(args.keyboard_backend).lower() == "evdev":
                # 全局热键：无需窗口在前台
                if evdev_keys.get("q", False):
                    evdev_keys["q"] = False
                    running = False
                    break
                if evdev_keys.get("r", False):
                    evdev_keys["r"] = False
                    recording = not recording
                    if recording:
                        if recorder.create_episode():
                            step_count = 0
                            print("✅ episode recording started")
                        else:
                            recording = False
                    else:
                        recorder.save_episode()
                        print("✅ episode saving triggered")

            if recording:
                data_dict: Dict[str, Any] = {"idx": step_count}
                data_dict["rgb"] = rgb
                data_dict["t_img"] = now_ms()
                data_dict["t_record_ms"] = now_ms()

                # optional: save realsense depth into json (no image file)
                if args.vision_backend == "realsense" and args.rs_depth and hasattr(vision, "get_depth"):
                    depth = vision.get_depth()  # type: ignore
                    data_dict["depth"] = depth.tolist() if depth is not None else None

                # Batch GET from Redis
                try:
                    for k in redis_keys:
                        redis_pipeline.get(k)
                    results = redis_pipeline.execute()

                    for raw, dk in zip(results, data_dict_keys):
                        data_dict[dk] = safe_json_loads(raw)
                except Exception as e:
                    # Skip this frame but keep loop alive
                    print(f"⚠️ Redis read error: {e}")
                    continue

                recorder.add_item(data_dict)
                step_count += 1

                elapsed = time.time() - t0
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
            else:
                # Not recording: avoid busy loop
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, exiting...")
    finally:
        try:
            if evdev_listener is not None:
                evdev_listener.stop()
        except Exception:
            pass
        try:
            if recording:
                recorder.save_episode()
        except Exception:
            pass

        try:
            recorder.close()
        except Exception:
            pass

        try:
            if vision is not None:
                vision.close()
        except Exception:
            pass

        try:
            if window_enabled:
                cv2.destroyAllWindows()
        except Exception:
            pass

        print(f"\nDone! Episodes saved under: {task_dir}")


if __name__ == "__main__":
    main()


