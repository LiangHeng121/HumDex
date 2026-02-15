import cv2
import threading

class RealTimeCamera:
    def __init__(self, device_id=0, width=1280, height=720, fps=30):
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # 有的后端支持缓冲区大小设置，可以试一下
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.latest_frame = None
        self.running = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """后台线程：不停读取最新帧，旧的都扔掉"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.latest_frame = frame

    def read(self):
        """主线程：永远返回“最近一次拿到的那帧”"""
        return self.latest_frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()
