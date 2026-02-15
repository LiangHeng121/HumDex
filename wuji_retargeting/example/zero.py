import time
import numpy as np
import wujihandpy

def read_current_pose(hand):
    qpos = hand.get_joint_actual_position()
    print("\n当前关节位置 (20):\n", qpos)
    return qpos

def go_to_pose(handcontroller, target_qpos, duration=1.0, steps=100):
    target_qpos = target_qpos.reshape(5,4)
    current = hand.get_joint_actual_position()   # ✅ 用 hand 读

    for t in np.linspace(0, 1, steps):
        interp = current * (1 - t) + target_qpos * t
        handcontroller.set_joint_target_position(interp)
        time.sleep(duration / steps)

    print("\n已到达目标姿态。")

if __name__ == "__main__":
    hand = wujihandpy.Hand()

    hand.write_joint_enabled(True)
    time.sleep(0.3)

    print("读取当前零位...")
    zero_pose = read_current_pose(hand)

    print("\n3 秒后回到该零位")
    time.sleep(3)

    controller = hand.realtime_controller(
        enable_upstream=False,
        filter=wujihandpy.filter.LowPass(cutoff_freq=10.0)   # ← 必须写
    )

    go_to_pose(controller, zero_pose, duration=1.0, steps=100)

    controller.close()

    hand.write_joint_enabled(False)
