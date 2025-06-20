import cv2
import numpy as np

# Camera resolution
width, height = 3264, 2448
display_width, display_height = 640, 480  # Downscale for display

# Intrinsic camera matrix (from calibration)
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

# Distance between the cameras (stereo baseline in meters)
baseline = 0.095  # 📏 <--- This is the distance between the two cameras (in meters)

# Open both cameras
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Remap matrices
map1_0, map2_0 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_16SC2)
map1_1, map2_1 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_16SC2)

# Blue color range in HSV
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

def get_block_position(frame, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            return (cx, cy), (x, y, w, h)
    return None, None

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

        # Detect blue object
        pos0, rect0 = get_block_position(view0, lower_blue, upper_blue)
        pos1, rect1 = get_block_position(view1, lower_blue, upper_blue)

        if pos0 and pos1:
            x0, y0 = pos0
            x1, y1 = pos1
            disparity = abs(x0 - x1)

            if disparity > 0:
                # Adjust focal length to match display resolution
                fx_scaled = new_K[0, 0] * (display_width / width)
                
                # Stereo depth in meters
                depth_m = (fx_scaled * baseline) / disparity
                depth_cm = depth_m * 100

                # Bounding box on left view
                x, y, w, h = rect0

                # Real-world size (width/height) in cm
                object_width_cm = (w * depth_m) / fx_scaled * 100
                object_height_cm = (h * depth_m) / fx_scaled * 100

                # Draw and annotate
                cv2.rectangle(view0, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(view0, f"Distance: {depth_cm:.1f} cm", (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(view0, f"W: {object_width_cm:.1f} cm  H: {object_height_cm:.1f} cm", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                
        # Combine views horizontally
        combined = np.hstack((view0, view1))
        cv2.imshow("Stereo Undistorted - Left | Right", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
