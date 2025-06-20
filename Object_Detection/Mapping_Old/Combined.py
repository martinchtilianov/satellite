import cv2
import numpy as np

# High-res input size (adjust to your camera's full resolution)
width, height = 3264, 2448

# Final display size (downscaled for visualization)
display_width, display_height = 960, 720

# Intrinsic camera matrix (from calibration)
K = np.array([[755.1285977,    0.0,          352.34985663],
              [0.0,            776.54558672, 210.00661258],
              [0.0,            0.0,            1.0]])

# Distortion coefficients (from calibration)
D = np.array([[ 0.16305384],
              [-1.38149094],
              [-0.03104714],
              [ 0.00644363],
              [ 3.62730084]])

# Open the camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Undistortion settings
scaling_factor = 2.0  # Increase this to zoom and flatten the distortion

# Modify camera matrix to zoom in and flatten fisheye effect
new_K = K.copy()
new_K[0, 0] *= scaling_factor  # fx
new_K[1, 1] *= scaling_factor  # fy

# Generate the remap matrices
map1, map2 = cv2.initUndistortRectifyMap(
    K, D, None, new_K, (width, height), cv2.CV_16SC2
)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Undistort frame
        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Resize both images for display
        frame_resized = cv2.resize(frame, (display_width, display_height))
        undist_resized = cv2.resize(undistorted, (display_width, display_height))

        # Show in two separate windows
        cv2.imshow("Original Feed", frame_resized)
        cv2.imshow("Undistorted Feed", undist_resized)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
