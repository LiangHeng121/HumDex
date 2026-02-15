import zmq
import cv2
import numpy as np

# 机器人的IP
ROBOT_IP = "192.168.123.164"
PORT = "5555"

context = zmq.Context()
socket = context.socket(zmq.SUB)
# 连接到机器人的 ZMQ 端口
socket.connect(f"tcp://{ROBOT_IP}:{PORT}")
socket.setsockopt_string(zmq.SUBSCRIBE, '')

print(f"Listening to {ROBOT_IP}:{PORT} ...")

while True:
    try:
        # 接收数据
        buffer = socket.recv()
        # 将字节流解码为图像 (假设发送的是编码后的帧或原始Buffer)
        # 注意：如果机器人发送的是 H264 裸流，OpenCV 无法直接这样解码，需要更复杂的解码器
        # 这里假设发送的是 JPEG 或者 OpenCV 支持的格式，或者你可以用来测试是否连通
        np_arr = np.frombuffer(buffer, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is not None:
            print(32342)
            cv2.imshow("Robot View", img)
        else:
            print("Received data but failed to decode image (Might be H264 raw stream)")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(e)