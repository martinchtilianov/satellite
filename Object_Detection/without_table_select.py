import cv2
import numpy as np

def load_calibration(calib_file):
    """
    Load stereo calibration data from a .npz file.
    """
    data = np.load(calib_file)
    left_map1  = data['left_map1']
    left_map2  = data['left_map2']
    right_map1 = data['right_map1']
    right_map2 = data['right_map2']
    Q = data['Q']
    return left_map1, left_map2, right_map1, right_map2, Q

def create_stereo_matcher(min_disp=0, num_disp=128, block_size=5):
    """
    Create a StereoSGBM matcher with given parameters.
    """
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return stereo

def detect_table_edge(rectL, mask_closed, points3D):
    """
    Detect the edge of the table in the rectified left image, using the provided mask,
    then compute the 3D distance from the camera to the closest point on that edge.
    Returns:
      - line_coords: (x1, y, x2, y) endpoints of the detected horizontal edge in image coords
      - closest_pt:  (X, Y, Z) coordinates of the closest 3D point on that edge
      - distance:    Euclidean distance (in meters) from camera center to closest_pt
    If no valid contour or 3D point is found, returns (None, None, None).
    """
    # 1) Find contours in the mask
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    # 2) Pick the largest contour (assumed to be the tabletop)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 5000:  # too small → probably noise
        return None, None, None

    # 3) Bounding rectangle of that contour
    x, y, w, h = cv2.boundingRect(largest)

    # 4) The top edge of this rectangle is where the table "drops off" → our edge.
    y_edge = y
    x1, y1 = x, y_edge
    x2, y2 = x + w, y_edge

    # 5) Reproject each pixel along that edge into 3D and pick the closest valid point
    edge_pts_3D = []
    H, W = points3D.shape[:2]
    for xi in range(x1, x2 + 1):
        yi = y_edge
        if 0 <= xi < W and 0 <= yi < H:
            X, Y, Z = points3D[yi, xi]
            if Z > 0 and Z < 10.0:
                edge_pts_3D.append((X, Y, Z))

    if not edge_pts_3D:
        return (x1, y1, x2, y2), None, None

    pts3d_array = np.array(edge_pts_3D)
    dists = np.linalg.norm(pts3d_array, axis=1)
    idx_min = np.argmin(dists)
    closest_point = tuple(pts3d_array[idx_min])
    distance = float(dists[idx_min])

    return (x1, y1, x2, y2), closest_point, distance

def compute_birds_eye(rectL, mask_closed, points3D, scale=200):
    """
    Using the mask & 3D points, make a bird's-eye view of the tabletop.
    Returns a color BEV image, or None if no valid points.
    """
    # 1) Find all pixel‐coordinates where mask_closed > 0
    ys, xs = np.where(mask_closed > 0)
    if len(xs) == 0:
        return None

    # 2) Gather their 3D positions
    all3D = points3D[ys, xs]
    valid = (all3D[:, 2] > 0) & (all3D[:, 2] < 10.0)
    if not np.any(valid):
        return None

    coords3D = all3D[valid]
    xs_valid = xs[valid]
    ys_valid = ys[valid]

    # 3) For BEV, we only care about X (left/right) and Z (forward)
    Xs = coords3D[:, 0]
    Zs = coords3D[:, 2]

    # 4) Determine min/max so we know how big to make the BEV image
    Xmin, Xmax = float(Xs.min()), float(Xs.max())
    Zmin, Zmax = float(Zs.min()), float(Zs.max())

    bev_width  = int(np.ceil((Xmax - Xmin) * scale)) + 1
    bev_height = int(np.ceil((Zmax - Zmin) * scale)) + 1

    # 5) Initialize BEV to mid‐gray (so you can see if it's empty)
    bev = np.full((bev_height, bev_width, 3), 200, dtype=np.uint8)

    # 6) Sample colors from rectL at those masked pixel locations
    colors = rectL[ys_valid, xs_valid]

    # 7) Convert each (X,Z) into pixel coords on BEV
    us = ((Xs - Xmin) * scale).astype(np.int32)
    vs = ((Zs - Zmin) * scale).astype(np.int32)
    vs = bev_height - 1 - vs  # flip so "closer (smaller Z)" is at bottom

    # 8) Paint those pixels onto the BEV
    bev[vs, us] = colors

    return bev

def main():
    # 1) Load stereo calibration
    calib_file = 'stereo_calib_live.npz'
    left_map1, left_map2, right_map1, right_map2, Q = load_calibration(calib_file)

    # 2) Create StereoSGBM matcher
    stereo = create_stereo_matcher()

    # 3) Open camera streams
    capL = cv2.VideoCapture(0)
    capR = cv2.VideoCapture(1)
    if not capL.isOpened() or not capR.isOpened():
        print("ERROR: Could not open camera streams.")
        return

    # 4) Make named windows
    cv2.namedWindow('Left Camera Raw')
    cv2.namedWindow('Right Camera Raw')
    cv2.namedWindow('Rectified Left + Edge')
    cv2.namedWindow('Disparity')
    cv2.namedWindow('Table Mask')
    cv2.namedWindow("Bird's Eye View")

    print("Press 'q' to quit. If 'Table Mask' is all black, try tweaking the thresholds below in code.")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("ERROR: Could not read from cameras.")
            break

        # 5) Show raw feeds
        cv2.imshow('Left Camera Raw', frameL)
        cv2.imshow('Right Camera Raw', frameR)

        # 6) Rectify both frames
        rectL = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, right_map1, right_map2, cv2.INTER_LINEAR)

        # 7) ===== TABLE MASKING STEP =====
        H, W = rectL.shape[:2]

        # 7a) We only look at the bottom 40% of the image (where the table lives)
        y_start = int(H * 0.60)
        roi = rectL.copy()
        roi[0:y_start, :] = 0  # zero out everything above row y_start

        # 7b) Convert to HSV and threshold for “off-white/light‐gray”
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # TWEAK THESE CONSTANTS if needed until “Table Mask” lights up the table:
        lower_white = np.array([0,  0,  150], dtype=np.uint8)  # H min=0, S min=0, V min=150
        upper_white = np.array([180, 60, 255], dtype=np.uint8)  # H max=180, S max=60, V max=255
        
        mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

        # 7c) Additionally, do a quick BGR‐based “light gray” threshold to catch any missed pixels
        lower_bgr = np.array([180, 180, 180], dtype=np.uint8)
        upper_bgr = np.array([255, 255, 255], dtype=np.uint8)
        mask_bgr = cv2.inRange(roi, lower_bgr, upper_bgr)

        # 7d) Combine HSV and BGR masks
        mask = cv2.bitwise_or(mask_hsv, mask_bgr)

        # 7e) Morphological closing to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 7f) Display the mask for debugging
        cv2.imshow('Table Mask', mask_closed)

        # 8) Compute disparity & reproject to 3D
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        disp16 = stereo.compute(grayL, grayR)
        disp = disp16.astype(np.float32) / 16.0
        points3D = cv2.reprojectImageTo3D(disp, Q)

        # 9) Detect edge and measure distance
        result = detect_table_edge(rectL, mask_closed, points3D)
        if result is not None:
            line_coords, closest_pt, distance = result
        else:
            line_coords, closest_pt, distance = (None, None, None)

        # 10) Draw on rectified left
        output = rectL.copy()
        if line_coords is not None:
            x1, y1, x2, y2 = line_coords
            # Draw detected edge in RED
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if closest_pt is not None:
            xm = int((x1 + x2)//2)
            ym = y1
            cv2.circle(output, (xm, ym), 5, (0, 255, 0), -1)
            cv2.putText(
                output,
                f"{distance:.2f} m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        cv2.imshow('Rectified Left + Edge', output)

        # 11) Display normalized disparity
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
        cv2.imshow('Disparity', disp_vis)

        # 12) Compute Bird’s‐Eye View
        bev = compute_birds_eye(rectL, mask_closed, points3D, scale=200)
        if bev is None:
            # If no BEV points, show a mid‐gray window with text
            gray_bev = np.full((300, 400, 3), 200, dtype=np.uint8)
            cv2.putText(gray_bev, "No BEV data", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Bird's Eye View", gray_bev)
        else:
            cv2.imshow("Bird's Eye View", bev)

        # 13) Print out counts for debugging (optional)
        num_masked = cv2.countNonZero(mask_closed)
        print(f"Masked pixels: {num_masked}", end='\r')

        # 14) Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
