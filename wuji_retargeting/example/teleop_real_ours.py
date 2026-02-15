import sys
from pathlib import Path
import time
import numpy as np
import wujihandpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wuji_retargeting import WujiHandRetargeter
from wuji_retargeting.mediapipe import apply_mediapipe_transformations


def smooth_move(hand, controller, target_qpos, duration=1.0, steps=100):
    """平滑移动到某个 5×4 的关节目标"""
    target_qpos = target_qpos.reshape(5, 4)
    cur = hand.get_joint_actual_position()

    for t in np.linspace(0, 1, steps):
        q = cur * (1 - t) + target_qpos * t
        controller.set_joint_target_position(q)
        time.sleep(duration / steps)


def run_teleop_npz(
    hand_side="right",
    npz_path="your_data.npz",
):
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}

    # ---------------------
    # Load NPZ
    # ---------------------
    data = np.load(npz_path, allow_pickle=True)
    track = data["hand_track_data"]    # list-like
    num_frames = len(track)

    # ---------------------
    # Initialize robot hand
    # ---------------------
    hand = wujihandpy.Hand()
    hand.write_joint_enabled(True)
    controller = hand.realtime_controller(
        enable_upstream=False,
        filter=wujihandpy.filter.LowPass(cutoff_freq=10.0)
    )
    time.sleep(0.5)

    # 记录启动时的零位（当前实际手的初始姿态）
    zero_pose = hand.get_joint_actual_position()
    print("\n记录启动零位：\n", zero_pose)

    # ---------------------
    # Initialize retargeter
    # ---------------------
    retargeter = WujiHandRetargeter(hand_side=hand_side)

    print("Start teleop from NPZ...")

    try:
        while True:    # 每一轮：播放完整 num_frames

            print("\n▶▶▶ 新一轮开始，共 {} 帧".format(num_frames))

            frame_idx = 0
            while frame_idx < num_frames:

                frame = track[frame_idx]
                kpts = frame["kpt_3ds_registered"][hand_side]   # (1,21,3)

                if isinstance(kpts, list) and len(kpts) == 0:
                    print("Empty keypoints, skip.")
                    frame_idx += 1
                    time.sleep(0.05)
                    continue

                mediapipe_pose = kpts[0]    # (21,3)
                print(mediapipe_pose)
                input("...")
                mediapipe_pose = apply_mediapipe_transformations(mediapipe_pose, hand_side)

                # ---- Retarget ----
                retarget_result = retargeter.retarget(mediapipe_pose)

                # ---- Control ----
                qpos = retarget_result.robot_qpos.reshape(5, 4)
                controller.set_joint_target_position(qpos)

                frame_idx += 1
                time.sleep(0.05)

            # ---------------------
            # 一轮结束 → 回到零位
            # ---------------------
            print("\n⏪ 所有帧已完成，回零位中...")
            smooth_move(hand, controller, zero_pose, duration=1.0)
            print("✔ 已回到零位。")

            # ---------------------
            # 等待用户按键
            # ---------------------
            input("\n按回车键开始下一轮播放...")

    except KeyboardInterrupt:
        print("\n用户停止。")

    finally:
        print("\n关闭控制器并失能电机...")
        controller.close()
        hand.write_joint_enabled(False)
        print("✔ 完成退出。")


if __name__ == "__main__":
    config = {
        "hand_side": "left",
        "npz_path": "/home/heng/heng/wuji/wuji_retargeting/example/processing_outputs.npz",
    }
    run_teleop_npz(**config)
