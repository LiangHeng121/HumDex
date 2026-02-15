import cv2

def open_camera(device_id=0):
    cap = cv2.VideoCapture(device_id)

    # 设置分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("摄像头已打开 /dev/video0")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取帧")
            break

        # 选择你需要的画面修复方式：
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # 1. 旋转 180°
        # frame = cv2.flip(frame, 1)                  # 2. 左右镜像
        # frame = cv2.flip(frame, 0)                # 3. 上下翻转

        cv2.imshow("USB Camera", frame)

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera(0)
