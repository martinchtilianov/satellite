import cv2
import numpy as np

# ==================== Configuration ====================
WIDTH, HEIGHT = 3264, 2448
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480

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

SCALING_FACTOR = 2.0
BASELINE_M = 0.06       # meters
NUM_DISPARITIES = 64
BLOCK_SIZE = 15
MIN_DISPARITY = 1       # pixels
DEPTH_FALLBACK_CM = 30
MIN_DISP_PIXELS = 50

MAP_SIZE = 500          # px
MAP_SCALE = 2           # px/cm

LARGE_CUBE_SIDE_CM = 5.0
SMALL_CUBE_SIDE_CM = 3.0
SIZE_THRESHOLD_CM = (LARGE_CUBE_SIDE_CM + SMALL_CUBE_SIDE_CM) / 2.0

# Now we keep just three color‐entries. For red, we store two HSV‐subranges (wrap‐around).
COLOR_RANGES = {
    'blue': (
        np.array([ 92,  93, 185], dtype=np.uint8),
        np.array([122, 234, 255], dtype=np.uint8),
        (255, 0, 0)       # BGR for blue‐cube outline/fill
    ),
    'red':  (
        (np.array([  0, 120,  70], dtype=np.uint8), np.array([ 10, 255, 255], dtype=np.uint8)),
        (np.array([170, 120,  70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
        (0, 0, 255)       # BGR for red‐cube
    ),
    'green':(
        np.array([ 40,  70,  70], dtype=np.uint8),
        np.array([ 80, 255, 255], dtype=np.uint8),
        (0, 255, 0)       # BGR for green‐cube
    ),
}

TABLE_RANGE = (
    np.array([  0,   0, 200], dtype=np.uint8),
    np.array([180,  30, 255], dtype=np.uint8)
)

MIN_CUBE_AREA_PX  = 10
MAX_CUBE_AREA_PX  = 200_000
MIN_TABLE_AREA_PX = 1_500

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Global state for HSV sampling
hsv_samples = []
sampling_mode = False

# ==================== Utility Functions ====================
def build_scaled_intrinsics(K_orig, scale):
    K_scaled = K_orig.copy().astype(float)
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale
    return K_scaled

def compute_undistort_maps(K_orig, D, K_new, size):
    return cv2.initUndistortRectifyMap(K_orig, D, None, K_new, size, cv2.CV_16SC2)

def get_depth_from_contour(disparity, contour, fx_scaled, fallback=DEPTH_FALLBACK_CM, min_pixels=MIN_DISP_PIXELS):
    mask = np.zeros(disparity.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    vals = disparity[mask == 255]
    vals = vals[~np.isnan(vals)]
    if len(vals) < min_pixels:
        return fallback
    med_disp = np.median(vals)
    if med_disp > MIN_DISPARITY:
        return (fx_scaled * BASELINE_M) / med_disp * 100.0
    else:
        return fallback

def project_point_to_map(x_img, depth_cm, fx_scaled, cx_scaled):
    X_world = (x_img - cx_scaled) * depth_cm / fx_scaled
    Y_world = depth_cm
    map_x = int(MAP_SIZE // 2 + X_world * MAP_SCALE)
    map_y = int(MAP_SIZE - Y_world * MAP_SCALE)
    return map_x, map_y

def mouse_callback(event, x, y, flags, param):
    global hsv_samples, sampling_mode
    if sampling_mode and event == cv2.EVENT_LBUTTONDOWN:
        hsv_val = param[y, x]
        hsv_samples.append(hsv_val)
        print(f"Sampled HSV: {hsv_val}")

def update_hsv_range(samples):
    arr = np.array(samples, dtype=np.uint8)
    mn, mx = arr.min(axis=0).astype(int), arr.max(axis=0).astype(int)
    lower = np.clip(mn - 10, [0,0,0], [180,255,255])
    upper = np.clip(mx + 10, [0,0,0], [180,255,255])
    if upper[0] < lower[0]:
        lower[0] = max(0, upper[0] - 10)
    print(f"✅ HSV range fixed:\n  Lower: {lower}\n  Upper: {upper}")
    return lower.astype(np.uint8), upper.astype(np.uint8)

def create_color_masks(hsv_img):
    """
    Return:
      - color_masks: {'blue': mask_blue, 'red': mask_red, 'green': mask_green}
      - table_mask
    """
    # Blue mask (single range)
    lb, ub, _ = COLOR_RANGES['blue']
    mask_blue = cv2.inRange(hsv_img, lb, ub)

    # Red mask (wraparound => combine two subranges)
    (lr1, ur1), (lr2, ur2), _ = COLOR_RANGES['red']
    mask_r1 = cv2.inRange(hsv_img, lr1, ur1)
    mask_r2 = cv2.inRange(hsv_img, lr2, ur2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    # Green mask (single range)
    lg, ug, _ = COLOR_RANGES['green']
    mask_green = cv2.inRange(hsv_img, lg, ug)

    # Table mask
    lt, ut = TABLE_RANGE
    mask_table = cv2.inRange(hsv_img, lt, ut)

    # Morphological opening to remove small noise
    mask_blue  = cv2.morphologyEx(mask_blue,  cv2.MORPH_OPEN, KERNEL)
    mask_red   = cv2.morphologyEx(mask_red,   cv2.MORPH_OPEN, KERNEL)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, KERNEL)
    mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_OPEN, KERNEL)

    return {'blue': mask_blue, 'red': mask_red, 'green': mask_green}, mask_table

def find_cube_contours(color_masks):
    """
    Given color_masks={'blue', 'red', 'green'}, return a list of (cnt, label, color_bgr).
    """
    results = []
    for label in ('blue', 'red', 'green'):
        mask = color_masks[label]
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # Get the BGR color from COLOR_RANGES (for red we stored it as the 3rd element)
        if label == 'red':
            color_bgr = COLOR_RANGES['red'][2]
        else:
            color_bgr = COLOR_RANGES[label][2]

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if MIN_CUBE_AREA_PX <= area <= MAX_CUBE_AREA_PX:
                results.append((cnt, label, color_bgr))
    return results

def draw_table_on_map(map_img, mask_table, disparity, fx_scaled, cx_scaled):
    """
    Fill each sufficiently large table contour in white, then draw its border in black.
    """
    cnts = cv2.findContours(mask_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        if cv2.contourArea(cnt) < MIN_TABLE_AREA_PX:
            continue

        z_table = get_depth_from_contour(disparity, cnt, fx_scaled)
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
    For each cube, compute its depth, approximate side length, and draw a filled square + label.
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
    Draw a bounding box + text label on the resized camera view for each cube contour.
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

def main():
    global sampling_mode, hsv_samples

    # Build scaled intrinsics & calculate once
    new_K = build_scaled_intrinsics(K, SCALING_FACTOR)
    map1_0, map2_0 = compute_undistort_maps(K, D, new_K, (WIDTH, HEIGHT))
    map1_1, map2_1 = compute_undistort_maps(K, D, new_K, (WIDTH, HEIGHT))

    # Precompute scale ratio and scaled fx, cx
    SCALE_RATIO = DISPLAY_WIDTH / float(WIDTH)
    fx_scaled_const = new_K[0, 0] * SCALE_RATIO
    cx_scaled_const = new_K[0, 2] * SCALE_RATIO

    # Initialize stereo matcher
    stereo = cv2.StereoBM_create(numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE)

    # Open cameras
    cap_left  = cv2.VideoCapture(1)
    cap_right = cv2.VideoCapture(0)
    for cap in (cap_left, cap_right):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    cv2.namedWindow("Stereo Left")
    cv2.setMouseCallback("Stereo Left", mouse_callback)

    try:
        while True:
            ret0, frame0 = cap_left.read()
            ret1, frame1 = cap_right.read()
            if not ret0 or not ret1:
                print("Camera error")
                break

            # Undistort full resolution, then resize
            und0 = cv2.remap(frame0, map1_0, map2_0, cv2.INTER_LINEAR)
            und1 = cv2.remap(frame1, map1_1, map2_1, cv2.INTER_LINEAR)
            view0 = cv2.resize(und0, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            view1 = cv2.resize(und1, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # Compute disparity on the resized grayscale pair
            gray0 = cv2.cvtColor(view0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(view1, cv2.COLOR_BGR2GRAY)
            disp = stereo.compute(gray0, gray1).astype(np.float32) / 16.0
            disp[disp <= 0] = np.nan

            # Use precomputed fx_scaled_const, cx_scaled_const
            fx_scaled = fx_scaled_const
            cx_scaled = cx_scaled_const

            # Generate masks
            hsv = cv2.cvtColor(view0, cv2.COLOR_BGR2HSV)
            if not sampling_mode:
                color_masks, mask_table = create_color_masks(hsv)
            else:
                # If in sampling mode, create empty masks so nothing is detected
                empty = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH), dtype=np.uint8)
                color_masks = {'blue': empty, 'red': empty, 'green': empty}
                mask_table = empty

            # ---- Draw table on top-down map ----
            map_img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
            draw_table_on_map(map_img, mask_table, disp, fx_scaled, cx_scaled)

            # ---- Detect cubes & draw both on map and on view ----
            cube_contours = find_cube_contours(color_masks)
            draw_cubes_on_map(map_img, cube_contours, disp, fx_scaled, cx_scaled)
            draw_cubes_on_view(view0, cube_contours, disp, fx_scaled, cx_scaled)

            # ---- Display windows ----
            combined_view = np.hstack((view0, view1))
            cv2.imshow("Stereo Left", combined_view)
            cv2.imshow("Top-Down Map", map_img)
            cv2.imshow("Blue Mask", color_masks['blue'])
            cv2.imshow("Red Mask", color_masks['red'])
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
                # Update only the blue range (for example). If you want to sample green or red,
                # you’d need to modify create_color_masks accordingly.
                COLOR_RANGES['blue'] = (lower, upper, COLOR_RANGES['blue'][2])
                sampling_mode = False
                print("🟢 Sampling mode OFF")

    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
