import cv2
import numpy as np

# ==================== Configuration ====================
# Camera resolution (original)
WIDTH, HEIGHT = 3264, 2448

# Display resolution (for processing & visualization)
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480

# Intrinsics
K = np.array([
    [755.1285977,   0.0,          352.34985663],
    [0.0,           776.54558672, 210.00661258],
    [0.0,             0.0,           1.0       ]
])
D = np.array([
    [ 0.16305384],
    [-1.38149094],
    [-0.03104714],
    [ 0.00644363],
    [ 3.62730084]
])

# Undistortion scaling
SCALING_FACTOR = 2.0

# Stereo setup
BASELINE_M = 0.06  # meters between cameras
NUM_DISPARITIES = 64
BLOCK_SIZE = 15
MIN_DISPARITY = 1

# Map parameters
MAP_SIZE = 500        # size of square bird’s-eye map image
MAP_SCALE = 2         # pixels per cm in map

# Cube sizes (cm)
LARGE_CUBE_SIDE_CM = 5.0
SMALL_CUBE_SIDE_CM = 3.0
SIZE_THRESHOLD_CM = (LARGE_CUBE_SIDE_CM + SMALL_CUBE_SIDE_CM) / 2.0

# HSV thresholds
COLOR_RANGES = {
    'blue':  (np.array([ 92,  93, 185]), np.array([122, 234, 255]), (255, 0,   0)),
    'red1':  (np.array([  0, 120,  70]), np.array([ 10, 255, 255]), None),
    'red2':  (np.array([170, 120,  70]), np.array([180, 255, 255]), None),
    'green': (np.array([ 40,  70,  70]), np.array([ 80, 255, 255]), (0,   255, 0)),
}
TABLE_RANGE = (np.array([0,   0, 200]), np.array([180, 30, 255]))

# Area thresholds (px)
MIN_CUBE_AREA_PX  = 10
MAX_CUBE_AREA_PX  = 200_000
MIN_TABLE_AREA_PX = 1_500

# Morphology kernel
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ==================== Global State for HSV Sampling ====================
hsv_samples = []
sampling_mode = False

# ==================== Utility Functions ====================
def build_scaled_intrinsics(K_orig, scale):
    """Return a new camera matrix scaled by 'scale'."""
    K_scaled = K_orig.copy().astype(float)
    K_scaled[0, 0] *= scale  # fx
    K_scaled[1, 1] *= scale  # fy
    K_scaled[0, 2] *= scale  # cx
    K_scaled[1, 2] *= scale  # cy
    return K_scaled

def compute_undistort_maps(K_orig, D, K_new, size):
    """Precompute remap maps for undistortion."""
    return cv2.initUndistortRectifyMap(
        K_orig, D, None, K_new, size, cv2.CV_16SC2
    )

def get_depth_from_contour(disparity, contour, fx_scaled, fallback=30, min_pixels=50):
    """
    Compute median disparity within 'contour' on the disparity map.
    Convert to depth in cm; if too few valid pixels or disparity <= MIN_DISPARITY, return fallback.
    """
    mask = np.zeros(disparity.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    vals = disparity[mask == 255]
    vals = vals[~np.isnan(vals)]
    if len(vals) < min_pixels:
        # Not enough disparity pixels: fallback
        return fallback
    med_disp = np.median(vals)
    if med_disp > MIN_DISPARITY:
        # Depth (cm) = (fx_scaled * baseline_m) / disp * 100
        return (fx_scaled * BASELINE_M) / med_disp * 100
    else:
        return fallback

def project_point_to_map(x_img, depth_cm, fx_scaled, cx_scaled):
    """
    Project a single image x-coordinate (x_img) at depth (cm) into bird’s-eye map coords.
    Returns (map_x, map_y).
    """
    X_world = (x_img - cx_scaled) * depth_cm / fx_scaled
    Y_world = depth_cm
    map_x = int(MAP_SIZE // 2 + X_world * MAP_SCALE)
    map_y = int(MAP_SIZE - Y_world * MAP_SCALE)
    return map_x, map_y

def mouse_callback(event, x, y, flags, param):
    """
    If sampling_mode is True and user clicks, record HSV at (x,y).
    """
    global hsv_samples, sampling_mode
    if sampling_mode and event == cv2.EVENT_LBUTTONDOWN:
        hsv_val = param[y, x]
        hsv_samples.append(hsv_val)
        print(f"Sampled HSV: {hsv_val}")

def update_hsv_range(samples):
    """Given a list of HSV samples (np.uint8), compute new lower/upper bounds with ±10 margin."""
    arr = np.array(samples, dtype=np.uint8)
    mn, mx = arr.min(axis=0).astype(int), arr.max(axis=0).astype(int)
    lower = np.clip(mn - 10,  [0,  0,  0], [180, 255, 255])
    upper = np.clip(mx + 10, [0,  0,  0], [180, 255, 255])
    # Handle hue wrap
    if upper[0] < lower[0]:
        lower[0] = max(0, upper[0] - 10)
    print(f"✅ HSV range fixed:\nLower: {lower}\nUpper: {upper}")
    return lower.astype(np.uint8), upper.astype(np.uint8)

def create_color_masks(hsv_img):
    """
    Given an HSV image, return:
      - masks: {'blue': mask_blue, 'red': mask_red, 'green': mask_green}
      - table_mask
    """
    # Blue mask
    lb, ub, _ = COLOR_RANGES['blue']
    mask_blue = cv2.inRange(hsv_img, lb, ub)

    # Red mask (combine two ranges)
    l1, u1, _ = COLOR_RANGES['red1']
    l2, u2, _ = COLOR_RANGES['red2']
    mask_red = cv2.bitwise_or(cv2.inRange(hsv_img, l1, u1),
                              cv2.inRange(hsv_img, l2, u2))

    # Green mask
    lg, ug, _ = COLOR_RANGES['green']
    mask_green = cv2.inRange(hsv_img, lg, ug)

    # Table mask
    lt, ut = TABLE_RANGE
    mask_table = cv2.inRange(hsv_img, lt, ut)

    # Morphological opening to remove noise
    mask_blue  = cv2.morphologyEx(mask_blue,  cv2.MORPH_OPEN, KERNEL)
    mask_red   = cv2.morphologyEx(mask_red,   cv2.MORPH_OPEN, KERNEL)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, KERNEL)
    mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_OPEN, KERNEL)

    return {'blue': mask_blue, 'red': mask_red, 'green': mask_green}, mask_table

def find_cube_contours(color_masks):
    """
    Given color_masks dict, return a list of (contour, label, color_bgr).
    """
    results = []
    for label in ('blue', 'red', 'green'):
        mask = color_masks[label]
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        _, _, color_bgr = COLOR_RANGES[label] if label != 'red' else COLOR_RANGES['red1']
        # For red, we just use the BGR from either red1 or red2; we'll pick (0,0,255).
        if label == 'red':
            color_bgr = (0, 0, 255)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if MIN_CUBE_AREA_PX <= area <= MAX_CUBE_AREA_PX:
                results.append((cnt, label, color_bgr))
    return results

def draw_table_on_map(map_img, mask_table, disparity, fx_scaled, cx_scaled):
    """
    Fill the table region in white and draw its border in black on map_img.
    """
    cnts = cv2.findContours(mask_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        if cv2.contourArea(cnt) < MIN_TABLE_AREA_PX:
            continue

        # Estimate depth once for the whole table
        z_table = get_depth_from_contour(disparity, cnt, fx_scaled)

        # Project every contour point into map coords
        pts_img = cnt.reshape(-1, 2)
        proj_pts = []
        for (x_img, _) in pts_img:
            mx, my = project_point_to_map(x_img, z_table, fx_scaled, cx_scaled)
            if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                proj_pts.append([mx, my])

        if len(proj_pts) < 3:
            continue

        proj_cnt = np.array(proj_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(map_img, [proj_cnt], (255, 255, 255))
        cv2.polylines(map_img, [proj_cnt], isClosed=True, color=(0, 0, 0), thickness=1)

def draw_cubes_on_map(map_img, cube_contours, disparity, fx_scaled, cx_scaled):
    """
    For each cube contour, compute depth, approximate size, and draw a colored square + label.
    """
    for cnt, label, color in cube_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx_img = x + w // 2
        z_cm = get_depth_from_contour(disparity, cnt, fx_scaled)
        size_cm = (w * z_cm) / fx_scaled
        side_cm = LARGE_CUBE_SIDE_CM if size_cm >= SIZE_THRESHOLD_CM else SMALL_CUBE_SIDE_CM
        mx, my = project_point_to_map(cx_img, z_cm, fx_scaled, cx_scaled)

        if not (0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE):
            continue

        half = int((side_cm / 2) * MAP_SCALE)
        cv2.rectangle(map_img, (mx - half, my - half), (mx + half, my + half), color, -1)
        cv2.circle(map_img, (mx, my), 3, (0, 0, 0), -1)
        cv2.putText(
            map_img, f"{label} {z_cm:.1f}cm",
            (mx + 5, my - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )

def draw_cubes_on_view(view, cube_contours, disparity, fx_scaled, cx_scaled):
    """
    Draw bounding boxes and distance labels for cubes on the resized view.
    """
    for cnt, label, color in cube_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        z_cm = get_depth_from_contour(disparity, cnt, fx_scaled)
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            view, f"{label} {z_cm:.1f}cm",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

# ==================== Main ====================
def main():
    global sampling_mode, hsv_samples

    # Build scaled intrinsics & undistort maps
    new_K = build_scaled_intrinsics(K, SCALING_FACTOR)
    map1_0, map2_0 = compute_undistort_maps(K, D, new_K, (WIDTH, HEIGHT))
    map1_1, map2_1 = compute_undistort_maps(K, D, new_K, (WIDTH, HEIGHT))

    # Initialize stereo matcher
    stereo = cv2.StereoBM_create(numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE)

    # Open both cameras
    cap0 = cv2.VideoCapture(1)
    cap1 = cv2.VideoCapture(0)
    for cap in (cap0, cap1):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    cv2.namedWindow("Stereo Left")
    cv2.setMouseCallback("Stereo Left", mouse_callback)

    try:
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            if not ret0 or not ret1:
                print("Camera error")
                break

            # Undistort then resize
            und0 = cv2.remap(frame0, map1_0, map2_0, cv2.INTER_LINEAR)
            und1 = cv2.remap(frame1, map1_1, map2_1, cv2.INTER_LINEAR)
            view0 = cv2.resize(und0, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            view1 = cv2.resize(und1, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # Compute disparity on grayscale resized images
            gray0 = cv2.cvtColor(view0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(view1, cv2.COLOR_BGR2GRAY)
            disp = stereo.compute(gray0, gray1).astype(np.float32) / 16.0
            disp[disp <= 0] = np.nan

            # Compute scaled fx & cx for resized view
            fx_scaled = new_K[0, 0] * (DISPLAY_WIDTH / WIDTH)
            cx_scaled = new_K[0, 2] * (DISPLAY_WIDTH / WIDTH)

            # HSV & masks
            hsv = cv2.cvtColor(view0, cv2.COLOR_BGR2HSV)
            if not sampling_mode:
                color_masks, mask_table = create_color_masks(hsv)
            else:
                # If sampling, don't process any masks
                color_masks = {'blue': np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH), np.uint8),
                               'red':  np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH), np.uint8),
                               'green': np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH), np.uint8)}
                mask_table = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH), np.uint8)

            # ---- Table mapping ----
            map_img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
            draw_table_on_map(map_img, mask_table, disp, fx_scaled, cx_scaled)

            # ---- Cube detection & mapping ----
            # Combine red1+red2 masks into 'red'
            combined_masks = {
                'blue': color_masks['blue'],
                'red':  cv2.bitwise_or(color_masks['red1'] if 'red1' in color_masks else color_masks['red'],
                                       color_masks['red2'] if 'red2' in color_masks else color_masks['red']),
                'green': color_masks['green']
            }
            cube_contours = find_cube_contours({
                'blue': color_masks['blue'],
                'red':  cv2.bitwise_or(color_masks['red1'] if 'red1' in color_masks else color_masks['red'],
                                       color_masks['red2'] if 'red2' in color_masks else color_masks['red']),
                'green': color_masks['green']
            })

            draw_cubes_on_map(map_img, cube_contours, disp, fx_scaled, cx_scaled)
            draw_cubes_on_view(view0, cube_contours, disp, fx_scaled, cx_scaled)

            # ---- Display windows ----
            combined_view = np.hstack((view0, view1))
            cv2.imshow("Stereo Left", combined_view)
            cv2.imshow("Top-Down Map", map_img)
            cv2.imshow("Blue Mask", color_masks['blue'])
            cv2.imshow("Red Mask", combined_masks['red'])
            cv2.imshow("Green Mask", color_masks['green'])
            cv2.imshow("Table Mask", mask_table)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                hsv_samples.clear()
                sampling_mode = True
                print("🟡 Sampling mode ON")
            elif key == 13 and hsv_samples:
                lower, upper = update_hsv_range(hsv_samples)
                COLOR_RANGES['blue'] = (lower, upper, COLOR_RANGES['blue'][2])
                sampling_mode = False
                print("🟢 Sampling mode OFF")

    finally:
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
