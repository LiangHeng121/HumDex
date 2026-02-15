import cv2
import numpy as np
import requests
from io import BytesIO


def sam3d_infer_one_image(image_bgr, server_url="http://127.0.0.1:8010/infer"):
    """
    输入一张 BGR 图（OpenCV 格式），发送给远程 SAM3D 服务器，返回 npz 里的 dict。
    """
    # 编码成 JPEG，减小传输量
    success, encoded_img = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise RuntimeError("Failed to encode image")

    files = {
        "image": ("frame.jpg", encoded_img.tobytes(), "image/jpeg")
    }

    resp = requests.post(server_url, files=files, timeout=10)
    resp.raise_for_status()

    # npz 在内存中解码，不落盘
    buf = BytesIO(resp.content)
    data = np.load(buf, allow_pickle=True)

    # 判断有没有人
    if "no_person" in data.files:
        print("SAM3D: no person detected in this frame.")
        return None

    return {k: data[k] for k in data.files}


def demo_run_single_image():
    server_url = "http://106.117.208.103:8010/infer"  # 改成你的服务器 IP

    image_path = "test.png"  # 本地测试图片
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image:", image_path)
        return

    result = sam3d_infer_one_image(img, server_url=server_url)
    if result is None:
        return

    pred_keypoints_3d = result["pred_keypoints_3d"]  # shape = (70, 3)

    print("pred_keypoints_3d shape:", pred_keypoints_3d.shape)
    # print("bbox:", result["bbox"])

    # 举个例子：提取左手 / 右手关节点
    # 参照你刚才发的 mhr70 配置：
    # 右手：21~41 (含 41)
    # 左手：42~62 (含 62)
    right_hand = pred_keypoints_3d[21:42]  # (21, 3)
    left_hand = pred_keypoints_3d[42:63]   # (21, 3)

    print("right hand:", right_hand.shape)
    print("left hand:", left_hand.shape)


if __name__ == "__main__":
    demo_run_single_image()
