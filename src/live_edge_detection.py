import av
import cv2


class VideoProcessorEdgeDetection:
    def __init__(self) -> None:
        self.threshold1 = 50
        self.threshold2 = 100
        self.flip = False
        self.canny = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.canny:
            img = cv2.Canny(img, self.threshold1, self.threshold2)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if self.flip:
            img = cv2.flip(img, 1)  # Left-Right flip

        return av.VideoFrame.from_ndarray(img, format="bgr24")
