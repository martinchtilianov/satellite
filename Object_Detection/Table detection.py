import cv2
import numpy as np

# -----------------------
# GLOBALS FOR ROI DRAWING
# -----------------------
roi_points = []
selection_complete = False
frame_for_selection = None

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback to record polygon points for ROI.
    Left-click to add points; right-click or 'n' to close polygon.
    """
    global roi_points, selection_complete, frame_for_selection

    if selection_complete:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        cv2.circle(frame_for_selection, (x, y), 3, (0, 255, 0), -1)
        if len(roi_points) > 1:
            cv2.line(frame_for_selection, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
        cv2.imshow("Select ROI - Press 'n' to finish", frame_for_selection)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(roi_points) > 2:
            selection_complete = True
            cv2.line(frame_for_selection, roi_points[-1], roi_points[0], (0, 255, 0), 2)
            cv2.imshow("Select ROI - Press 'n' to finish", frame_for_selection)

def create_trackbar_window(initial_lower, initial_upper):
    """
    Create trackbars to adjust HSV lower/upper thresholds.
    """
    cv2.namedWindow("HSV Controls", cv2.WINDOW_NORMAL)
    # Lower bounds
    cv2.createTrackbar("L - H", "HSV Controls", int(initial_lower[0]), 179, lambda x: None)
    cv2.createTrackbar("L - S", "HSV Controls", int(initial_lower[1]), 255, lambda x: None)
    cv2.createTrackbar("L - V", "HSV Controls", int(initial_lower[2]), 255, lambda x: None)
    # Upper bounds
    cv2.createTrackbar("U - H", "HSV Controls", int(initial_upper[0]), 179, lambda x: None)
    cv2.createTrackbar("U - S", "HSV Controls", int(initial_upper[1]), 255, lambda x: None)
    cv2.createTrackbar("U - V", "HSV Controls", int(initial_upper[2]), 255, lambda x: None)

def get_trackbar_values():
    """
    Read HSV lower/upper from the trackbars.
    """
    lh = cv2.getTrackbarPos("L - H", "HSV Controls")
    ls = cv2.getTrackbarPos("L - S", "HSV Controls")
    lv = cv2.getTrackbarPos("L - V", "HSV Controls")
    uh = cv2.getTrackbarPos("U - H", "HSV Controls")
    us = cv2.getTrackbarPos("U - S", "HSV Controls")
    uv = cv2.getTrackbarPos("U - V", "HSV Controls")
    lower = np.array([lh, ls, lv], dtype=np.uint8)
    upper = np.array([uh, us, uv], dtype=np.uint8)
    return lower, upper

def compute_initial_hsv_bounds(frame, polygon):
    """
    Given a BGR frame and polygon points, compute min/max HSV inside that region.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array([polygon], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == 255]
    if len(pixels) == 0:
        return np.array([0,0,0], dtype=np.uint8), np.array([179,255,255], dtype=np.uint8)
    h_vals = pixels[:, 0]
    s_vals = pixels[:, 1]
    v_vals = pixels[:, 2]
    lower = np.array([h_vals.min(), s_vals.min(), v_vals.min()], dtype=np.uint8)
    upper = np.array([h_vals.max(), s_vals.max(), v_vals.max()], dtype=np.uint8)
    return lower, upper

def main():
    global frame_for_selection, selection_complete

    # Load stereo rectification / Q from your previous step's .npz
    calib_file = 'stereo_calib_live.npz'
    data = np.load(calib_file)
    left_map1  = data['left_map1']
    left_map2  = data['left_map2']
    right_map1 = data['right_map1']
    right_map2 = data['right_map2']
    Q = data['Q']

    # Open cameras
    capL = cv2.VideoCapture(0)
    capR = cv2.VideoCapture(1)
    if not capL.isOpened() or not capR.isOpened():
        print("ERROR: Cannot open cameras.")
        return

    # Grab one left-frame for ROI selection
    retL, first_frame = capL.read()
    if not retL:
        print("ERROR: Cannot grab first frame.")
        return

    frame_for_selection = first_frame.copy()
    cv2.namedWindow("Select ROI - Press 'n' to finish", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select ROI - Press 'n' to finish", mouse_callback)

    print(">>> Draw a polygon around the tabletop (left-click to add points).")
    print(">>> Right-click or press 'n' (with ≥3 points) to finalize ROI.")
    cv2.imshow("Select ROI - Press 'n' to finish", frame_for_selection)

    # Wait for ROI selection
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n') and len(roi_points) > 2:
            selection_complete = True
            cv2.line(frame_for_selection, roi_points[-1], roi_points[0], (0, 255, 0), 2)
            cv2.imshow("Select ROI - Press 'n' to finish", frame_for_selection)
            break
        elif key == ord('q'):
            print("ROI selection canceled.")
            capL.release()
            capR.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Select ROI - Press 'n' to finish")

    # Compute HSV bounds from that polygon
    initial_lower, initial_upper = compute_initial_hsv_bounds(first_frame, roi_points)
    create_trackbar_window(initial_lower, initial_upper)

    # Create StereoSGBM for disparity
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=5,
        P1=8*3*5**2, P2=32*3*5**2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Windows for display
    cv2.namedWindow('Original Left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Original Right', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask Preview', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Rectified Left with Table Edge', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Birds Eye View', cv2.WINDOW_NORMAL)

    # Parameters for bird's eye view mapping
    bev_width, bev_height = 500, 500
    x_range = (-2.0, 2.0)   # Meters left/right range
    z_range = (0.0, 4.0)    # Meters forward range

    print("Adjust HSV sliders. Press 'q' to quit.")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("ERROR: Camera read failed.")
            break

        # Show raw camera feeds
        cv2.imshow('Original Left', frameL)
        cv2.imshow('Original Right', frameR)

        # Rectify
        rectL = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, right_map1, right_map2, cv2.INTER_LINEAR)

        # Get HSV thresholds
        lower, upper = get_trackbar_values()

        # Mask preview
        hsv_rect = cv2.cvtColor(rectL, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_rect, lower, upper)
        mask_vis = cv2.bitwise_and(rectL, rectL, mask=mask)
        cv2.imshow('Mask Preview', mask_vis)

        # Disparity → 3D
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        disp16 = stereo.compute(grayL, grayR)
        disp = disp16.astype(np.float32) / 16.0
        points3D = cv2.reprojectImageTo3D(disp, Q)

        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Contours → largest → bounding rect → table edge
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            y_edge = y
            x1, y1 = x, y_edge
            x2, y2 = x + w, y_edge

            # Reproject edge line into 3D, find closest
            edge_pts_3D = []
            H, W = points3D.shape[:2]
            for xi in range(x1, x2 + 1):
                yi = y_edge
                if 0 <= xi < W and 0 <= yi < H:
                    X, Y, Z = points3D[yi, xi]
                    if Z > 0 and Z < z_range[1]:
                        edge_pts_3D.append((X, Y, Z))
            if edge_pts_3D:
                pts3d_arr = np.array(edge_pts_3D)
                dists = np.linalg.norm(pts3d_arr, axis=1)
                idx_min = np.argmin(dists)
                closest_pt = tuple(pts3d_arr[idx_min])
                distance = dists[idx_min]
            else:
                closest_pt = None
                distance = None
        else:
            x1 = y1 = x2 = y2 = None
            closest_pt = None
            distance = None

        # Draw overlay on rectified left
        output = rectL.copy()
        if x1 is not None:
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if closest_pt is not None:
                xm = int((x1 + x2) / 2)
                ym = y1
                cv2.circle(output, (xm, ym), 5, (0, 255, 0), -1)
                cv2.putText(output, f"{distance:.2f} m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Rectified Left with Table Edge', output)

        # Show normalized disparity
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
        cv2.imshow('Disparity', disp_vis)

        # Build Bird's Eye View of table region
        bev = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        ys, xs = np.where(mask_closed > 0)
        H, W = points3D.shape[:2]
        for (u, v) in zip(xs, ys):
            if 0 <= u < W and 0 <= v < H:
                X, Y, Z = points3D[v, u]
                if Z > z_range[0] and Z < z_range[1] and X > x_range[0] and X < x_range[1]:
                    bx = int((X - x_range[0]) / (x_range[1] - x_range[0]) * bev_width)
                    bz = int((1 - (Z - z_range[0]) / (z_range[1] - z_range[0])) * bev_height)
                    if 0 <= bx < bev_width and 0 <= bz < bev_height:
                        bev[bz, bx] = (0, 255, 0)  # green for table

        # Mark camera at bottom center
        cam_x = int((0 - x_range[0]) / (x_range[1] - x_range[0]) * bev_width)
        cam_z = int((1 - (0 - z_range[0]) / (z_range[1] - z_range[0])) * bev_height)
        if 0 <= cam_x < bev_width and 0 <= cam_z < bev_height:
            cv2.circle(bev, (cam_x, cam_z), 5, (0, 0, 255), -1)

        cv2.imshow('Birds Eye View', bev)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
