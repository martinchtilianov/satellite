import cv2
import numpy as np

def load_calibration(calib_file):
    """
    Load stereo calibration data from a .npz file.
    """
    data = np.load(calib_file)
    return data['left_map1'], data['left_map2'], data['right_map1'], data['right_map2'], data['Q']

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
    Detect the edge of the table using the flood-fill mask and 3D points.
    Combines mask bounding‐rect with edge information to find the topmost horizontal line.
    Returns:
      - line_coords: (x1, y, x2, y) endpoints of the detected horizontal table edge
      - closest_pt:  (X, Y, Z) of the closest 3D point on that edge
      - distance:    Euclidean distance to that closest point
    If no valid contour or 3D point is found, returns (None, None, None).
    """
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 5000:
        return None, None, None

    x, y, w, h = cv2.boundingRect(largest)
    y_edge = y
    x1, y1 = x, y_edge
    x2, y2 = x + w, y_edge

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
    Make a bird's-eye view (top-down) of the flood-filled table region.
    """
    ys, xs = np.where(mask_closed > 0)
    if len(xs) == 0:
        return None

    all3D = points3D[ys, xs]
    valid = (all3D[:, 2] > 0) & (all3D[:, 2] < 10.0)
    if not np.any(valid):
        return None

    coords3D = all3D[valid]
    xs_valid = xs[valid]
    ys_valid = ys[valid]

    Xs = coords3D[:, 0]
    Zs = coords3D[:, 2]

    Xmin, Xmax = float(Xs.min()), float(Xs.max())
    Zmin, Zmax = float(Zs.min()), float(Zs.max())

    bev_width  = int(np.ceil((Xmax - Xmin) * scale)) + 1
    bev_height = int(np.ceil((Zmax - Zmin) * scale)) + 1

    bev = np.full((bev_height, bev_width, 3), 200, dtype=np.uint8)

    colors = rectL[ys_valid, xs_valid]
    us = ((Xs - Xmin) * scale).astype(np.int32)
    vs = ((Zs - Zmin) * scale).astype(np.int32)
    vs = bev_height - 1 - vs

    bev[vs, us] = colors
    return bev

def main():
    # 1) Load stereo calibration (replace path if needed)
    left_map1, left_map2, right_map1, right_map2, Q = load_calibration('stereo_calib_live.npz')

    # 2) Create the StereoSGBM matcher
    stereo = create_stereo_matcher()

    # 3) Open camera streams
    capL = cv2.VideoCapture(0)
    capR = cv2.VideoCapture(1)
    if not capL.isOpened() or not capR.isOpened():
        print("ERROR: Cannot open cameras.")
        return

    # 4) Create display windows
    cv2.namedWindow('Left Camera Raw')
    cv2.namedWindow('Right Camera Raw')
    cv2.namedWindow('Rectified Left + Edge')
    cv2.namedWindow('Disparity')
    cv2.namedWindow('Table Mask')
    cv2.namedWindow('Table Edges')
    cv2.namedWindow("Bird's Eye View")

    print("Press 'q' to quit. Adjust loDiff/hiDiff if 'Table Mask' is empty.")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("ERROR: Cannot read frames.")
            break

        # Show raw camera feeds
        cv2.imshow('Left Camera Raw', frameL)
        cv2.imshow('Right Camera Raw', frameR)

        # 5) Rectify the frames
        rectL = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, right_map1, right_map2, cv2.INTER_LINEAR)

        # 6) Flood‐fill segmentation on rectified left
        H, W = rectL.shape[:2]
        flood_mask = np.zeros((H + 2, W + 2), np.uint8)
        seed_point = (W - 1, H - 1)        # Bottom‐right corner (on the table)
        lo_diff = (30, 30, 30)             # Increase tolerance if needed
        hi_diff = (30, 30, 30)
        temp = rectL.copy()               # Copy for floodFill color sampling
        cv2.floodFill(
            temp,
            flood_mask,
            seed_point,
            newVal=(255,255,255),
            loDiff=lo_diff,
            upDiff=hi_diff,
            flags=cv2.FLOODFILL_MASK_ONLY
        )

        # Extract the valid mask (remove 1-pixel border)
        table_mask = flood_mask[1:-1, 1:-1]
        # Morphological closing to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask_closed = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)

        # Display the table mask
        mask_vis = cv2.cvtColor((mask_closed * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imshow('Table Mask', mask_vis)

        # 7) Canny edge detection within the masked region
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        # Blur first to reduce noise
        blurred = cv2.GaussianBlur(grayL, (5, 5), 0)
        # Lowered thresholds (30,80) to pick up faint edges
        edges_full = cv2.Canny(blurred, 30, 80)
        # Keep only edges inside the table mask
        edges_in_table = cv2.bitwise_and(edges_full, edges_full, mask=mask_closed.astype(np.uint8))
        cv2.imshow('Table Edges', edges_in_table)

        # 8) Compute disparity and reproject to 3D
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        disp16 = stereo.compute(grayL, grayR)
        disp = disp16.astype(np.float32) / 16.0
        points3D = cv2.reprojectImageTo3D(disp, Q)

        # 9) Detect table edge (horizontal top of bounding rect)
        result = detect_table_edge(rectL, mask_closed, points3D)
        if result is not None:
            line_coords, closest_pt, distance = result
        else:
            line_coords, closest_pt, distance = (None, None, None)

        # 10) Draw everything on the rectified left image
        output = rectL.copy()
        # Overlay the in‐table edges in blue
        colored_edges = cv2.cvtColor(edges_in_table, cv2.COLOR_GRAY2BGR)
        colored_edges[:, :, 2] = 255  # make them bright blue
        overlay = cv2.addWeighted(output, 0.8, colored_edges, 0.2, 0)
        output = overlay

        if line_coords is not None:
            x1, y1, x2, y2 = line_coords
            # Red line for the detected table edge
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if closest_pt is not None:
            xm = int((x1 + x2) // 2)
            ym = y1
            cv2.circle(output, (xm, ym), 6, (0, 255, 0), -1)
            cv2.putText(
                output,
                f"{distance:.2f} m",
                (x1, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        cv2.imshow('Rectified Left + Edge', output)

        # 11) Display normalized disparity
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
        cv2.imshow('Disparity', disp_vis)

        # 12) Bird’s‐Eye View of masked region
        bev = compute_birds_eye(rectL, mask_closed, points3D, scale=200)
        if bev is None:
            gray_bev = np.full((300, 400, 3), 200, dtype=np.uint8)
            cv2.putText(
                gray_bev,
                "No BEV data",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("Bird's Eye View", gray_bev)
        else:
            cv2.imshow("Bird's Eye View", bev)

        # 13) Debugging: print how many pixels are masked
        num_masked = cv2.countNonZero(mask_closed)
        print(f"Masked pixels: {num_masked}", end='\r')

        # 14) Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
