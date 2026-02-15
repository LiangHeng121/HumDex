#!/usr/bin/env python3
"""
Standalone ZMQ vision receiver + viewer (for quick testing).

Compatible with deploy_real/server_realsense_zmq_pub.py message format:
    [int32 width][int32 height][int32 jpeg_length][jpeg_bytes]

Usage (on laptop):
  source ~/miniconda3/bin/activate twist2
  cd deploy_real
  python test_zmq_vision_sub_viewer.py --ip 192.168.123.164 --port 5555
"""

import argparse
import struct
import time

import cv2
import numpy as np
import zmq


def parse_args():
    p = argparse.ArgumentParser(description="ZMQ SUB viewer for TWIST2 vision stream (JPEG).")
    p.add_argument("--ip", required=True, help="Vision server IP (e.g. g1 ip)")
    p.add_argument("--port", default=5555, type=int, help="ZMQ port (default: 5555)")
    p.add_argument("--window", default="ZMQ Vision Viewer", help="OpenCV window name")
    p.add_argument("--timeout_ms", default=2000, type=int, help="Poll timeout (ms)")
    p.add_argument("--print_every", default=60, type=int, help="Print stats every N frames")
    return p.parse_args()


def main():
    args = parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{args.ip}:{args.port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    frame = 0
    last_t = time.time()
    last_print = last_t

    try:
        while True:
            events = dict(poller.poll(args.timeout_ms))
            if sock not in events:
                print(f"[viewer] timeout {args.timeout_ms}ms: no data from tcp://{args.ip}:{args.port}")
                continue

            msg = sock.recv()
            if len(msg) < 12:
                continue

            w = struct.unpack("i", msg[0:4])[0]
            h = struct.unpack("i", msg[4:8])[0]
            jpeg_len = struct.unpack("i", msg[8:12])[0]
            jpeg = msg[12:]
            if len(jpeg) != jpeg_len:
                continue

            img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            frame += 1
            now = time.time()
            if args.print_every > 0 and frame % args.print_every == 0:
                fps = args.print_every / (now - last_print)
                last_print = now
                print(f"[viewer] frames={frame} fps={fps:.1f} header={w}x{h} decoded={img.shape[1]}x{img.shape[0]} jpeg={jpeg_len/1024:.1f}KB")

            cv2.imshow(args.window, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            sock.close(0)
            ctx.term()
        except Exception:
            pass


if __name__ == "__main__":
    main()


