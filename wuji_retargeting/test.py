import cv2
import numpy as np
import requests
from io import BytesIO
import time


def sam3d_infer_one_image(image_bgr, server_url):
    """发送图像并返回 npz dict"""
    success, encoded_img = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise RuntimeError("Failed to encode image")

    files = {
        "image": ("frame.jpg", encoded_img.tobytes(), "image/jpeg")
    }

    resp = requests.post(server_url, files=files, timeout=10)
    resp.raise_for_status()

    buf = BytesIO(resp.content)
    data = np.load(buf, allow_pickle=True)

    return {k: data[k] for k in data.files}


def benchmark_fps(server_url="http://106.117.208.103:8010/infer",
                  image_path="test.png",
                  warmup=3,
                  num_iters=50):
    """测服务器 FPS 和延迟"""

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    print("开始 warmup ...")
    for _ in range(warmup):
        sam3d_infer_one_image(img, server_url)

    print("开始正式测量...\n")

    times = []
    for i in range(num_iters):
        t0 = time.time()
        _ = sam3d_infer_one_image(img, server_url)
        t1 = time.time()

        elapsed = t1 - t0
        times.append(elapsed)

        print(f"[{i+1:02d}/{num_iters}] 单帧耗时: {elapsed*1000:.2f} ms")

    avg = sum(times) / len(times)
    fps = 1.0 / avg

    print("\n================== SAM3D Benchmark ==================")
    print(f"平均单帧耗时: {avg*1000:.2f} ms")
    print(f"最快: {min(times)*1000:.2f} ms")
    print(f"最慢: {max(times)*1000:.2f} ms")
    print(f"平均 FPS: {fps:.2f}")
    print("=====================================================\n")


if __name__ == "__main__":
    benchmark_fps(
        server_url="http://106.117.208.103:8010/infer",
        image_path="test.png",
        warmup=5,
        num_iters=50
    )
