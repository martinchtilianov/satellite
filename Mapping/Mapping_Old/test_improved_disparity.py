import cv2
import numpy as np

# ==================== Camera & Calibration ====================
width, height = 3264, 2448
display_width, display_height = 640, 480

# Original intrinsics
K = np.array([[755.1285977,   0.0,          352.34985663],
              [0.0,           776.54558672, 210.00661258],
              [0.0,             0.0,           1.0       ]])
D = np.array([[ 0.16305384],
              [-1.38149094],
              [-0.03104714],
              [ 0.00644363],
              [ 3.62730084]])

# Build a “scaled” K for undistortion (including the principal point!)
scaling_factor = 2.0
new_K = K.copy()
new_K[0,0] *= scaling_factor   # fx
new_K[1,1] *= scaling_factor   # fy
new_K[0,2] *= scaling_factor   # cx
new_K[1,2] *= scaling_factor   # cy

baseline = 0.04  # meters

# Open cameras
cap0 = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(0)
for cap in (cap0, cap1):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Precompute undistort maps
map1_0, map2_0 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_16SC2)
map1_1, map2_1 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_16SC2)

# ==================== Mapping & Detection Setup ====================
map_size      = 500
map_scale     = 2
min_disp      = 1    # disparity below this is unreliable
stereo        = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# HSV thresholds
lower_blue    = np.array([ 92,  93, 185])
upper_blue    = np.array([122, 234, 255])
lower_table   = np.array([  0,   0, 100])
upper_table   = np.array([180,  60, 255])

# Area thresholds (tweak these to reject more/less noise)
MIN_CUBE_AREA  = 500     # minimum #px for a blue blob to be considered a cube
MAX_CUBE_AREA  = 200000  # maximum
MIN_TABLE_AREA = 1500    # minimum #px for a table contour

# Morphology kernel (to remove tiny speckles)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# HSV sampling
hsv_samples   = []
sampling_mode = False

# ==================== Utility Functions ====================
def get_depth_from_contour(disparity, contour, fx_scaled,
                           fallback=30, min_pixels=50):
    # 1) draw the filled contour mask
    mask = np.zeros(disparity.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # 2) dilate to grab more edge pixels
    dilate_k = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, dilate_k, iterations=1)

    # 3) collect valid (non-NaN) disparity values
    vals = disparity[mask == 255]
    vals = vals[~np.isnan(vals)]

    # 4) if too few samples, bail out
    if len(vals) < min_pixels:
        print(f"⚠️ Only {len(vals)} valid disparity pixels (<{min_pixels}). Using fallback.")
        return fallback

    # 5) compute median disparity
    m = np.median(vals)
    print(f"✅ Median disparity: {m:.2f}")

    # 6) if still too small, fallback
    if m <= min_disp:
        print("⚠️ Disparity too small. Using fallback.")
        return fallback

    # 7) triangulate to depth in cm
    return (fx_scaled * baseline) / m * 100

def project_to_map(x_img, depth_cm, fx_scaled, cx_scaled):
    # subtract true center, not display_width/2
    X_world = (x_img - cx_scaled) * depth_cm / fx_scaled
    Y_world = depth_cm
    map_x = int(map_size//2 + X_world * map_scale)
    map_y = int(map_size    - Y_world * map_scale)
    return map_x, map_y

def mouse_callback(event, x, y, flags, param):
    global hsv_samples, sampling_mode
    if sampling_mode and event == cv2.EVENT_LBUTTONDOWN:
        hsv_samples.append(param[y, x])
        print(f"Sampled HSV: {param[y, x]}")

def update_hsv_range(samples):
    arr = np.array(samples, dtype=np.uint8)
    mn, mx = arr.min(axis=0).astype(int), arr.max(axis=0).astype(int)
    lower = np.clip(mn - 10, [0,0,0], [180,255,255])
    upper = np.clip(mx + 10, [0,0,0], [180,255,255])
    if upper[0] < lower[0]: upper[0] = lower[0] + 10
    if upper[2] < lower[2]: upper[2] = lower[2] + 10
    print(f"✅ HSV range fixed:\nLower: {lower}\nUpper: {upper}")
    return lower.astype(np.uint8), upper.astype(np.uint8)

# ==================== Main Loop ====================
cv2.namedWindow("Stereo Left")
cv2.setMouseCallback("Stereo Left", mouse_callback, param=None)

try:
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            print("Camera error")
            break

        # Undistort & resize
        und0  = cv2.remap(frame0, map1_0, map2_0, cv2.INTER_LINEAR)
        und1  = cv2.remap(frame1, map1_1, map2_1, cv2.INTER_LINEAR)
        view0 = cv2.resize(und0, (display_width, display_height))
        view1 = cv2.resize(und1, (display_width, display_height))

        # Disparity
        gray0 = cv2.cvtColor(view0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(view1, cv2.COLOR_BGR2GRAY)
        disp  = stereo.compute(gray0, gray1).astype(np.float32) / 16.0
        disp[disp <= 0] = np.nan

        # Scaled intrinsics
        fx_scaled = new_K[0,0] * (display_width / width)
        cx_scaled = new_K[0,2] * (display_width / width)

        # HSV masking (or sampling)
        hsv = cv2.cvtColor(view0, cv2.COLOR_BGR2HSV)
        cv2.setMouseCallback("Stereo Left", mouse_callback, param=hsv)
        if not sampling_mode:
            mask_blue  = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_table = cv2.inRange(hsv, lower_table, upper_table)
        else:
            mask_blue  = np.zeros_like(hsv[:,:,0])
            mask_table = np.zeros_like(hsv[:,:,0])

        # Morphological opening
        mask_blue  = cv2.morphologyEx(mask_blue,  cv2.MORPH_OPEN, KERNEL)
        mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_OPEN, KERNEL)

        # Find cube contours
        conts_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        for c in conts_blue:
            area = cv2.contourArea(c)
            if len(c)<3 or area==0:
                cv2.drawContours(view0, [c], -1, (0,255,255),1)
            elif MIN_CUBE_AREA <= area <= MAX_CUBE_AREA:
                cv2.drawContours(view0, [c], -1, (0,255,0),1)
                valid_contours.append(c)

        # Build table map
        conts_table, _ = cv2.findContours(mask_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        for c in conts_table:
            if cv2.contourArea(c) < MIN_TABLE_AREA:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            z  = get_depth_from_contour(disp, c, fx_scaled)
            mx, my = project_to_map(cx, z, fx_scaled, cx_scaled)
            if 0<=mx<map_size and 0<=my<map_size:
                cv2.circle(map_img, (mx,my), 1, (255,255,255), -1)

        # Draw cubes on map & annotate
        for c in valid_contours:
            area = cv2.contourArea(c)
            print(f"Detected cube contour area: {area}")
            x,y,w,h = cv2.boundingRect(c)
            cx_img = x + w//2

            z = get_depth_from_contour(disp, c, fx_scaled)
            mx, my = project_to_map(cx_img, z, fx_scaled, cx_scaled)
            if not (0<=mx<map_size and 0<=my<map_size):
                continue

            size_px = int(5 * map_scale)
            cv2.rectangle(map_img,
                          (mx-size_px, my-size_px),
                          (mx+size_px, my+size_px),
                          (255,0,0), -1)
            cv2.circle(map_img, (mx,my), 3, (0,0,255), -1)
            cv2.line(map_img, (map_size//2, map_size), (mx,my), (0,0,255),1)
            cv2.putText(map_img, f"{z:.1f} cm", (mx+5, my-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200),1)

            cv2.putText(view0, f"{z:.1f}cm", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),2)
            cv2.rectangle(view0, (x,y), (x+w,y+h), (255,0,0),2)
            if z == 30:
                cv2.putText(view0, "↳ fallback", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,100,255),1)

        # Show recent HSV samples
        for i, val in enumerate(hsv_samples[-5:]):
            cv2.putText(view0, f"HSV: {val}", (10, 20+20*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,255,0),1)

        # Display
        combined = np.hstack((view0, view1))
        cv2.imshow("Stereo Left", combined)
        cv2.imshow("Top-Down Map", map_img)
        cv2.imshow("Blue Mask", mask_blue)
        cv2.imshow("Table Mask", mask_table)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            hsv_samples.clear()
            sampling_mode = True
            print("🟡 Sampling mode: Click the cube in 'Stereo Left'")
        elif key == 13 and hsv_samples:
            lower_blue, upper_blue = update_hsv_range(hsv_samples)
            sampling_mode = False

finally:
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
