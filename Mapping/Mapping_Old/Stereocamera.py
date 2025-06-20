import cv2
import numpy as np

# Camera resolution
width, height = 3264, 2448
display_width, display_height = 640, 480  # downscale for performance

# Intrinsic camera matrix
K = np.array([[755.1285977, 0.0, 352.34985663],
              [0.0, 776.54558672, 210.00661258],
              [0.0, 0.0, 1.0]])

# Distortion coefficients
D = np.array([[0.16305384],
              [-1.38149094],
              [-0.03104714],
              [0.00644363],
              [3.62730084]])

# Modify camera matrix to zoom
scaling_factor = 2.0
new_K = K.copy()
new_K[0, 0] *= scaling_factor
new_K[1, 1] *= scaling_factor

# Open both cameras
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Generate remap maps
map1_0, map2_0 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_16SC2)
map1_1, map2_1 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_16SC2)

# Blue color range in HSV (tweak as needed)
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

def detect_blue_block(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # minimum area threshold
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Blue Block", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame

try:
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            print("Failed to capture from one or both cameras")
            break

        # Undistort both
        undist0 = cv2.remap(frame0, map1_0, map2_0, interpolation=cv2.INTER_LINEAR)
        undist1 = cv2.remap(frame1, map1_1, map2_1, interpolation=cv2.INTER_LINEAR)

        # Resize for display
        view0 = cv2.resize(undist0, (display_width, display_height))
        view1 = cv2.resize(undist1, (display_width, display_height))

        # Detect blue block in both views
        view0 = detect_blue_block(view0)
        view1 = detect_blue_block(view1)

        # Combine into one window
        combined = np.hstack((view0, view1))
        cv2.imshow("Undistorted Views - Camera 0 (Left) | Camera 1 (Right)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
