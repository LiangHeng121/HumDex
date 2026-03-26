#!/usr/bin/env python3
"""Local execution client for G1 real-world chunk-blocking test.

Run this on local machine where humdex controllers are running.
It collects local obs (vision + redis state), requests action chunks from
DreamZero server, publishes actions to Redis, and uploads executed vision
frames back to server for server-side recording/merging.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import uuid
from collections import deque

import cv2
import numpy as np
import redis
import websockets.sync.client
import zmq
from openpi_client import msgpack_numpy


def now_ms() -> int:
    return int(time.time() * 1000)


class ZmqJpegSubscriber:
    def __init__(self, host: str, port: int, timeout_ms: int = 50) -> None:
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(f"tcp://{host}:{port}")
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)
        self.timeout_ms = int(timeout_ms)
        self.last_frame: np.ndarray | None = None

    def get_frame(self) -> np.ndarray:
        events = dict(self.poller.poll(self.timeout_ms))
        if self.sock in events:
            msg = self.sock.recv()
            if len(msg) >= 12:
                width = struct.unpack("i", msg[0:4])[0]
                height = struct.unpack("i", msg[4:8])[0]
                jpeg_len = struct.unpack("i", msg[8:12])[0]
                payload = msg[12:]
                if len(payload) == jpeg_len:
                    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if bgr is not None and bgr.shape[0] == height and bgr.shape[1] == width:
                        self.last_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    elif bgr is not None:
                        self.last_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.last_frame is None:
            self.last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return self.last_frame.copy()

    def close(self) -> None:
        self.sock.close()
        self.ctx.term()


class G1RedisIO:
    def __init__(self, redis_ip: str, robot_key: str) -> None:
        self.redis = redis.Redis(host=redis_ip, port=6379, decode_responses=False)
        self.redis.ping()
        self.robot_key = robot_key
        self.key_state_body = f"state_body_{robot_key}"
        self.key_state_l = f"state_wuji_hand_left_{robot_key}"
        self.key_state_r = f"state_wuji_hand_right_{robot_key}"
        self.key_action_body = f"action_body_{robot_key}"
        self.key_action_l = f"action_wuji_qpos_target_left_{robot_key}"
        self.key_action_r = f"action_wuji_qpos_target_right_{robot_key}"
        self.key_t_action = "t_action"
        self.key_t_action_l = f"t_action_wuji_hand_left_{robot_key}"
        self.key_t_action_r = f"t_action_wuji_hand_right_{robot_key}"
        self.key_mode_l = f"wuji_hand_mode_left_{robot_key}"
        self.key_mode_r = f"wuji_hand_mode_right_{robot_key}"

    def _get_json_vec(self, key: str, dim: int) -> np.ndarray:
        raw = self.redis.get(key)
        if raw is None:
            return np.zeros((dim,), dtype=np.float64)
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            arr = np.asarray(json.loads(raw), dtype=np.float64).reshape(-1)
        except Exception:
            return np.zeros((dim,), dtype=np.float64)
        if arr.shape[0] < dim:
            out = np.zeros((dim,), dtype=np.float64)
            out[: arr.shape[0]] = arr
            return out
        return arr[:dim]

    def read_state71(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        body_raw = self._get_json_vec(self.key_state_body, 34)
        body31 = body_raw[:31]
        left20 = self._get_json_vec(self.key_state_l, 20)
        right20 = self._get_json_vec(self.key_state_r, 20)
        return body31, left20, right20

    def publish_action(self, action75: np.ndarray) -> None:
        a = np.asarray(action75, dtype=np.float32).reshape(-1)
        body = a[:35].tolist()
        left = a[35:55].tolist()
        right = a[55:75].tolist()
        ts = now_ms()
        p = self.redis.pipeline()
        p.set(self.key_action_body, json.dumps(body))
        p.set(self.key_action_l, json.dumps(left))
        p.set(self.key_action_r, json.dumps(right))
        p.set(self.key_t_action, ts)
        p.set(self.key_t_action_l, ts)
        p.set(self.key_t_action_r, ts)
        p.set(self.key_mode_l, "follow")
        p.set(self.key_mode_r, "follow")
        p.execute()


class WsClient:
    def __init__(self, host: str, port: int) -> None:
        self.uri = f"ws://{host}:{port}"
        self.packer = msgpack_numpy.Packer()
        self.ws = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            ping_interval=60,
            ping_timeout=600,
        )
        self.meta = msgpack_numpy.unpackb(self.ws.recv())
        print(f"[WS] connected: {self.uri} meta={self.meta}")

    def request_infer(self, session_id: str, prompt: str, obs: dict) -> dict:
        req = {"endpoint": "infer", "session_id": session_id, "prompt": prompt, "obs": obs}
        self.ws.send(self.packer.pack(req))
        resp = self.ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(resp)
        return msgpack_numpy.unpackb(resp)

    def log_chunk(self, session_id: str, chunk_id: int, vision_frames: np.ndarray, executed_actions: np.ndarray) -> dict:
        req = {
            "endpoint": "log_chunk",
            "session_id": session_id,
            "chunk_id": int(chunk_id),
            "vision_frames": vision_frames.astype(np.uint8),
            "executed_actions": executed_actions.astype(np.float32),
        }
        self.ws.send(self.packer.pack(req))
        resp = self.ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(resp)
        return msgpack_numpy.unpackb(resp)

    def end(self, session_id: str) -> dict:
        req = {"endpoint": "end", "session_id": session_id}
        self.ws.send(self.packer.pack(req))
        resp = self.ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(resp)
        return msgpack_numpy.unpackb(resp)

    def close(self) -> None:
        self.ws.close()


def _make_obs(frame_buf: deque[np.ndarray], body31: np.ndarray, left20: np.ndarray, right20: np.ndarray, prompt: str, obs_frames: int) -> dict:
    frames = list(frame_buf)[-obs_frames:]
    if len(frames) == 0:
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
    while len(frames) < obs_frames:
        frames.insert(0, frames[0])
    video = np.stack(frames, axis=0).astype(np.uint8)
    obs = {
        "video.head": video,
        "state.body_core": body31.reshape(1, -1).astype(np.float64),
        "state.wuji_hand_left": left20.reshape(1, -1).astype(np.float64),
        "state.wuji_hand_right": right20.reshape(1, -1).astype(np.float64),
        "annotation.task": prompt,
    }
    return obs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local G1 execution client (chunk blocking)")
    p.add_argument("--server_host", required=True)
    p.add_argument("--server_port", type=int, default=8000)
    p.add_argument("--redis_ip", default="localhost")
    p.add_argument("--robot_key", default="unitree_g1_with_hands")
    p.add_argument("--vision_host", default="127.0.0.1")
    p.add_argument("--vision_port", type=int, default=5555)
    p.add_argument("--obs_frames", type=int, default=1)
    p.add_argument("--control_fps", type=float, default=30.0)
    p.add_argument("--num_chunks", type=int, default=50)
    p.add_argument("--prompt", default="pick up the object")
    p.add_argument("--session_id", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    session_id = args.session_id.strip() or str(uuid.uuid4())
    print(f"[SESSION] {session_id}")
    dt = 1.0 / float(args.control_fps)

    vision = ZmqJpegSubscriber(args.vision_host, args.vision_port)
    redis_io = G1RedisIO(args.redis_ip, args.robot_key)
    ws = WsClient(args.server_host, args.server_port)
    frame_buf: deque[np.ndarray] = deque(maxlen=max(1, args.obs_frames))

    try:
        for chunk_i in range(args.num_chunks):
            frame_buf.append(vision.get_frame())
            body31, left20, right20 = redis_io.read_state71()
            obs = _make_obs(frame_buf, body31, left20, right20, args.prompt, args.obs_frames)
            infer_resp = ws.request_infer(session_id=session_id, prompt=args.prompt, obs=obs)
            if not infer_resp.get("ok", False):
                raise RuntimeError(f"infer failed: {infer_resp}")
            chunk_id = int(infer_resp["chunk_id"])
            action_chunk = np.asarray(infer_resp["action_chunk"], dtype=np.float32)
            H = int(action_chunk.shape[0])
            print(
                f"[CHUNK {chunk_i}] id={chunk_id} H={H} pred_video_frames={infer_resp.get('pred_video_frames')} "
                f"infer_sec={infer_resp.get('infer_sec', -1):.3f}"
            )

            vis_frames = []
            exec_actions = []
            for s in range(H):
                t0 = time.time()
                a = action_chunk[s]
                redis_io.publish_action(a)
                exec_actions.append(a)
                frame = vision.get_frame()
                frame_buf.append(frame)
                vis_frames.append(frame)
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            log_resp = ws.log_chunk(
                session_id=session_id,
                chunk_id=chunk_id,
                vision_frames=np.stack(vis_frames, axis=0),
                executed_actions=np.stack(exec_actions, axis=0),
            )
            if not log_resp.get("ok", False):
                raise RuntimeError(f"log_chunk failed: {log_resp}")
    except KeyboardInterrupt:
        print("[INTERRUPT] stopping...")
    finally:
        try:
            end_resp = ws.end(session_id=session_id)
            print(f"[END] {end_resp}")
        except Exception as e:
            print(f"[WARN] end failed: {e}")
        ws.close()
        vision.close()


if __name__ == "__main__":
    main()
