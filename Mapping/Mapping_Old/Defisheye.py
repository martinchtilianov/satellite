import cv2
import numpy as np
from defisheye import Defisheye
import os

class LiveDefisheye:
    def __init__(self, dtype, format, fov, pfov, resolution):
        self.dtype = dtype
        self.format = format
        self.fov = fov
        self.pfov = pfov
        self.resolution = resolution  

    def undistort(self, frame):
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        defish = Defisheye(
            infile=temp_path,
            dtype=self.dtype,
            format=self.format,
            fov=self.fov,
            pfov=self.pfov
        )
        result = defish.convert()
        os.remove(temp_path)
        return result

# Defisheye configuration
defisheye_processor = LiveDefisheye(
    dtype='linear',
    format='fullframe',  # changed from 'circular'
    fov=180,
    pfov=120,
    resolution=(640, 480)
)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        undistorted = defisheye_processor.undistort(frame)

        undistorted_resized = cv2.resize(undistorted, (frame.shape[1], frame.shape[0]))
        combined = np.hstack((frame, undistorted_resized))

        cv2.imshow("Original (Left) vs Undistorted (Right)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
