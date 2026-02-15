import cv2
import numpy as np
import requests
import time
from io import BytesIO
import sys
from pathlib import Path
import wujihandpy

from camera import RealTimeCamera

# ---------------------------
# WujiHand Retarget imports
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wuji_retargeting import WujiHandRetargeter
from wuji_retargeting.mediapipe import apply_mediapipe_transformations

# ---------------------------
# SAM3D → MediaPipe Mapping
# ---------------------------
SAM3D_TO_MEDIAPIPE = [
    20,
    3,2,1,0,
    7,6,5,4,
    11,10,9,8,
    15,14,13,12,
    19,18,17,16
]

def scale_finger_bones(mediapipe_pose, scales_per_bone):
    """
    mediapipe_pose: (21,3) mediapipe 格式
    scales_per_bone: dict:
        {
            "thumb": [s0, s1, s2, s3],
            "index": [s0, s1, s2, s3],
            "middle":[s0, ...],
            "ring": [...],
            "pinky":[...]
        }
    """
    pose = mediapipe_pose.copy()

    # mediapipe finger indices
    FINGERS = {
        "thumb":  [1,2,3,4],
        "index":  [5,6,7,8],
        "middle": [9,10,11,12],
        "ring":   [13,14,15,16],
        "pinky":  [17,18,19,20]
    }

    WRIST = pose[0]

    for finger_name, idxs in FINGERS.items():
        scales = scales_per_bone[finger_name]  # length 4
        assert len(scales) == 4

        # 逐段计算并缩放
        prev = WRIST
        for i, idx in enumerate(idxs):
            raw_vec = pose[idx] - prev        # 原始骨骼向量
            scaled_vec = raw_vec * scales[i]  # 逐段缩放
            pose[idx] = prev + scaled_vec     # 更新坐标
            prev = pose[idx]                  # 下一段的起点

    return pose


def show_resized(winname, img, max_width=1280):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(winname, img)


def smooth_move(hand, controller, target_qpos, duration=1.0, steps=100):
    """平滑移动到某个 5×4 的关节目标"""
    target_qpos = target_qpos.reshape(5, 4)
    cur = controller.get_joint_actual_position()

    for t in np.linspace(0, 1, steps):
        q = cur * (1 - t) + target_qpos * t
        controller.set_joint_target_position(q)
        time.sleep(duration / steps)


# ==========================================================
# 发送图片到 SAM3D 服务器（阻塞等待返回）
# ==========================================================
def sam3d_infer(image_bgr, server_url):
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        print("JPEG编码失败")
        return None

    try:
        resp = requests.post(
            server_url,
            files={"image": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
            timeout=10            # 连接3秒，推理最多等10秒
        )
        resp.raise_for_status()
    except Exception as e:
        print("❌ SAM3D 请求失败：", e)
        return None

    data = np.load(BytesIO(resp.content), allow_pickle=True)

    if "no_person" in data:
        print("⚠ SAM3D 未检测到人")
        return None

    return data


# ==========================================================
# 实时摄像头 + SAM3D + Skeleton 显示 + WujiHand
# ==========================================================
def run_realtime(hand_side="left", server_url="", cam_id=0):

    hand_side = hand_side.lower()
    assert hand_side in ["left", "right"]

    cam = RealTimeCamera(2)

    print("启动实时遥操作... 按 q 退出")

    # ----------------- 初始化 WujiHand -----------------
    hand = wujihandpy.Hand()
    hand.write_joint_enabled(True)
    controller = hand.realtime_controller(
        enable_upstream=True,
        filter=wujihandpy.filter.LowPass(cutoff_freq=10.0)
    )
    time.sleep(0.4)

    zero_pose = hand.get_joint_actual_position()
    retargeter = WujiHandRetargeter(hand_side)


    try:
        while True:
            frame = cam.read()
            # print("read")
            if frame is None:
                print("❌ 无法读取摄像头")
                continue
            # h, w = frame.shape[:2]
            # frame = frame[:, :w // 2]   # 左半部分


            # 上下反转
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            # cv2.imwrite("output2.png", frame)

            # -----------------------------
            # 推理：必须等待返回（不排队）
            # -----------------------------
            result = sam3d_infer(frame, server_url)
            if result is None:
                continue

            # 取 skeleton_image（HxWx3）
            skeleton = result["skeleton_image"].astype(np.uint8)

            # -----------------------------
            # 拼接显示 Camera + Skeleton
            # -----------------------------
            h = min(frame.shape[0], skeleton.shape[0])
            frame_resized = cv2.resize(frame, (frame.shape[1], h))
            skeleton_resized = cv2.resize(skeleton, (skeleton.shape[1], h))

            stack = np.hstack([frame_resized, skeleton_resized])
            # cv2.imshow("Camera + Skeleton", stack)
            cv2.imwrite("output.png", stack)
            # print("shown image")

            # -----------------------------
            # 提取关键点，执行 retarget
            # -----------------------------
            pred = result["pred_keypoints_3d"]

            if hand_side == "left":
                kpts = pred[42:63] - pred[62]
            else:
                kpts = pred[21:42] - pred[41]

            mediapipe_pose = kpts[SAM3D_TO_MEDIAPIPE]

            # 按你之前的补偿参数
            mediapipe_pose[1:5] *= 1.1
            mediapipe_pose[5:9] *= 1.14
            mediapipe_pose[9:13] *= 1.15
            mediapipe_pose[13:17] *= 1.1
            mediapipe_pose[17:21] *= 1.15

            # scales = {
            #     "thumb":  [1.0, 1.0, 1.0, 1.1],
            #     "index":  [1.0, 1.0, 1.0, 1.14],
            #     "middle": [1.0, 1.0, 1.0, 1.15],
            #     "ring":   [1.0, 1.0, 1.0, 1.1],
            #     "pinky":  [1.0, 1.0, 1.0, 1.15],
            # }

            # mediapipe_pose = scale_finger_bones(mediapipe_pose, scales)

            mediapipe_pose = apply_mediapipe_transformations(mediapipe_pose, hand_side)

            retarget_result = retargeter.retarget(mediapipe_pose)
            qpos = retarget_result.robot_qpos.reshape(5, 4)

            # controller.set_joint_target_position(qpos)
            smooth_move(hand, controller, qpos, duration=1.0, steps=100)

            # time.sleep(1.0)

            # print("teleoped")

    except KeyboardInterrupt:
        print("\n用户停止。")
        smooth_move(hand, controller, zero_pose, duration=1.0)
        print("✔ 已回到零位。")

    finally:
        print("\n关闭控制器并失能电机...")
        controller.close()
        hand.write_joint_enabled(False)
        cam.release()
        cv2.destroyAllWindows()
        print("✔ 完成退出。")

if __name__ == "__main__":
    run_realtime(
        hand_side="left",
        server_url="http://106.117.208.103:8010/infer",
        cam_id=0
    )
