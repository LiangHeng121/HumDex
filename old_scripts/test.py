# quickstart.py
import time
import wujihandpy

hand = wujihandpy.Hand()
try:
    # 使能所有关节（阻塞，保证成功）
    hand.write_joint_enabled(True)

    # 使食指第 0 关节转至 1.57 rad（约 90° 下压）
    hand.finger(1).joint(0).write_joint_target_position(1.57)

    # 等待动作完成
    time.sleep(0.5)
finally:
    # 使用完毕务必失能所有关节
    hand.write_joint_enabled(False)