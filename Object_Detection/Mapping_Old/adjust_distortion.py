import cv2
import numpy as np

# Camera resolution
width, height = 3264, 2448

# Estimated camera matrix
K = np.array([[width, 0, width / 2],
              [0, height, height / 2],
              [0, 0, 1]])

# Estimated distortion coefficients
D = np.array([[-0.3], [0.1], [0.0], [0.0]])

# Open the camera
cap = cv2.VideoCapture(1)  # Adjust index if necessary
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Prepare undistortion maps
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), np.eye(3), balance=0.0)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Resize for display
    display_frame = cv2.resize(frame, (960, 720))
    display_undistorted = cv2.resize(undistorted, (960, 720))
    combined = np.hstack((display_frame, display_undistorted))

    cv2.imshow("Original (Left) vs Undistorted (Right)", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
