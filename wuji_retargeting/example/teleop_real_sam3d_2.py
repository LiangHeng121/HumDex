import sys
from pathlib import Path
import time
import numpy as np
import wujihandpy
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wuji_retargeting import WujiHandRetargeter
from wuji_retargeting.mediapipe import apply_mediapipe_transformations

SAM3D_TO_MEDIAPIPE = [
    20,   # wrist

    3, 2, 1, 0,     # thumb: CMC, MCP, IP, TIP

    7, 6, 5, 4,     # index: MCP, PIP, DIP, TIP
    11, 10, 9, 8,   # middle
    15, 14, 13, 12, # ring
    19, 18, 17, 16  # pinky
]

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
    npz_folder_path="your_data_folder"
):
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}

    # ---------------------
    # 获取文件夹中的所有 .npz 文件
    # ---------------------
    npz_files = [f for f in os.listdir(npz_folder_path) if f.endswith('.npz')]
    npz_files.sort()  # 按文件名顺序排序

    # ---------------------
    # 初始化机器人手部控制器
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
    # 初始化 retargeter
    # ---------------------
    retargeter = WujiHandRetargeter(hand_side=hand_side)

    print("Start teleop from NPZ folder...")

    try:
        while True:  # 持续循环读取每一轮的所有 .npz 文件

            print(f"\n▶▶▶ 开始新一轮，处理文件夹中的所有 .npz 文件...")

            for npz_file in npz_files:
                print(f"\n正在处理文件: {npz_file}")

                # ---------------------
                # 加载每个 .npz 文件
                # ---------------------
                npz_path = os.path.join(npz_folder_path, npz_file)
                frame = np.load(npz_path, allow_pickle=True)

                if "person_0_pred_keypoints_3d" not in frame:
                    print("Empty keypoints, skip.")
                    time.sleep(0.05)
                    continue

                pred_keypoints_3d = frame["person_0_pred_keypoints_3d"]  # (70, 3)

                # 从pred_keypoints_3d中提取左手或右手关键点
                if hand_side == "left":
                    kpts = pred_keypoints_3d[42:63]  # 提取左手关键点 (19-39)
                else:
                    kpts = pred_keypoints_3d[21:42]  # 提取右手关键点 (40-60)

                # ---- SAM3D_TO_MEDIAPIPE 映射 ----
                mediapipe_pose = kpts[SAM3D_TO_MEDIAPIPE]  # 根据SAM3D_TO_MEDIAPIPE映射

                # print(mediapipe_pose - mediapipe_pose[0])

                # ---- 计算调整方向 ----
                wrist = mediapipe_pose[0]  # wrist是固定的
                finger_scales = np.array([1.0, 1.14, 1.076, 1.116, 1.21])  # 初始的scale参数，可以根据实际需要进行调整

                # ---- 对每个手指的4个点进行调整 ----
                adjusted_kpts = mediapipe_pose.copy()  # 深拷贝原始关键点数据
                for i, finger_id in enumerate([1, 2, 3, 4, 5]):  # 对每个手指进行调整
                    # 获取手指的起始点（wrist与指根之间的向量）
                    start_idx = 4 * (finger_id - 1) + 1
                    finger_points = mediapipe_pose[start_idx:start_idx+4]
                    vector = finger_points[-1] - wrist  # 计算手指的方向向量

                    # 沿着向量方向调整
                    for j in range(4):
                        finger_points[j] += vector * (finger_scales[i]-1)  # 使用每个手指不同的缩放因子

                    adjusted_kpts[start_idx:start_idx+4] = finger_points  # 更新手指点位置
                
                adjusted_kpts = apply_mediapipe_transformations(adjusted_kpts, hand_side)

                # ---- Retarget ----
                retarget_result = retargeter.retarget(adjusted_kpts)  # 重定向

                # ---- 控制手部 ----
                qpos = retarget_result.robot_qpos.reshape(5, 4)  # 获取机器人关节位置
                # input()
                controller.set_joint_target_position(qpos)  # 设置目标关节位置

                time.sleep(0.05)

            # ---------------------
            # 一轮结束 → 回到零位
            # ---------------------
            print("\n⏪ 所有帧已完成，回零位中...")
            smooth_move(hand, controller, zero_pose, duration=1.0)
            print("✔ 已回到零位。")

            # ---------------------
            # 等待用户按键开始下一轮
            # ---------------------
            input("\n按回车键开始下一轮播放...")

    except KeyboardInterrupt:
        print("\n用户停止。")
        smooth_move(hand, controller, zero_pose, duration=1.0)
        print("✔ 已回到零位。")

    finally:
        print("\n关闭控制器并失能电机...")
        controller.close()
        hand.write_joint_enabled(False)
        print("✔ 完成退出。")


if __name__ == "__main__":
    config = {
        "hand_side": "left",  # 选择控制左手或右手
        "npz_folder_path": "/home/heng/heng/example_output",  # 输入包含 .npz 文件的文件夹路径
    }
    run_teleop_npz(**config)  # 执行遥操作
