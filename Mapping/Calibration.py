import cv2
import numpy as np

# -----------------------
#  CONFIGURATION
# -----------------------

# Change these if your chessboard has a different number of inner corners:
BOARD_WIDTH  = 8   # number of inner corners along the width (columns)
BOARD_HEIGHT = 6   # number of inner corners along the height (rows)
BOARD_SIZE = (BOARD_WIDTH, BOARD_HEIGHT)

# Physical size of one square (in meters)
SQUARE_SIZE = 0.025  # e.g. 2.5 cm

# How many valid views to collect before stopping (per camera)
NUM_VIEWS = 15

# Indices of your two cameras (check with e.g. 0 and 1, or adjust if needed)
CAMERA_LEFT_IDX  = 0
CAMERA_RIGHT_IDX = 1

# File in which to save the final calibration data
OUTPUT_CALIB_FILE = 'stereo_calib_live.npz'


# -----------------------
#  PREPARE OBJECT POINTS
# -----------------------

# Create a template of object points: (0,0,0), (1,0,0), ... (BOARD_WIDTH-1,BOARD_HEIGHT-1,0), scaled by SQUARE_SIZE
objp = np.zeros((BOARD_HEIGHT * BOARD_WIDTH, 3), np.float32)
objp[:, :2] = np.indices((BOARD_WIDTH, BOARD_HEIGHT)).T.reshape(-1, 2)
objp *= SQUARE_SIZE


# -----------------------
#  CAPTURE LOOP
# -----------------------

# Lists to store the collected object points and image points for left/right
objpoints = []
imgpoints_left  = []
imgpoints_right = []

# Create windows
cv2.namedWindow('Left Camera')
cv2.namedWindow('Right Camera')

# Open both cameras
cap_left  = cv2.VideoCapture(CAMERA_LEFT_IDX)
cap_right = cv2.VideoCapture(CAMERA_RIGHT_IDX)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("ERROR: Could not open one or both camera indices.")
    exit(1)

print(f"Press 'c' when you see the full chessboard in both windows to capture a pair.")
print(f"Collect {NUM_VIEWS} valid pairs. Press 'q' to quit early.")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
views_collected = 0
img_shape = None

while views_collected < NUM_VIEWS:
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    if not retL or not retR:
        print("ERROR: Could not read frames from cameras.")
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Detect chessboard corners but do NOT refine/append yet
    ret_corners_L, cornersL = cv2.findChessboardCorners(grayL, BOARD_SIZE, None)
    ret_corners_R, cornersR = cv2.findChessboardCorners(grayR, BOARD_SIZE, None)

    # Draw detected corners (if found) for visual feedback
    visL = frameL.copy()
    visR = frameR.copy()
    if ret_corners_L:
        cv2.drawChessboardCorners(visL, BOARD_SIZE, cornersL, ret_corners_L)
    if ret_corners_R:
        cv2.drawChessboardCorners(visR, BOARD_SIZE, cornersR, ret_corners_R)

    cv2.imshow('Left Camera',  visL)
    cv2.imshow('Right Camera', visR)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting early.")
        break

    # When user presses 'c', and corners are found in BOTH images, capture the view
    if key == ord('c') and ret_corners_L and ret_corners_R:
        # Refine to subpixel accuracy
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        # Save the object points and the image points
        objpoints.append(objp.copy())
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

        # Record image size for calibration
        if img_shape is None:
            img_shape = (grayL.shape[1], grayL.shape[0])  # (width, height)

        views_collected += 1
        print(f"Captured view #{views_collected}")

        # Optionally, give a quick visual confirmation by showing the board in a window for 500ms
        cv2.imshow('Left Camera', visL)
        cv2.imshow('Right Camera', visR)
        cv2.waitKey(500)

print(f"Done capturing. Collected {views_collected} valid views.")

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

if views_collected < 1:
    print("Not enough valid views for calibration. Exiting.")
    exit(1)


# -----------------------
#  SINGLE-CAMERA CALIBRATION
# -----------------------

print("\nCalibrating left camera...")
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpoints_left, img_shape, None, None
)
print(f"  Left RMS reprojection error: {retL:.4f}")

print("\nCalibrating right camera...")
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpoints_right, img_shape, None, None
)
print(f"  Right RMS reprojection error: {retR:.4f}")


# -----------------------
#  STEREO CALIBRATION
# -----------------------

print("\nRunning stereo calibration (this may take a moment)...")
flags = cv2.CALIB_FIX_INTRINSIC  # keep individual intrinsics fixed

stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    mtxL, distL,
    mtxR, distR,
    img_shape,
    criteria=stereo_criteria,
    flags=flags
)
print(f"  Stereo RMS reprojection error: {retS:.4f}")
print("Rotation matrix between cameras (R):")
print(R)
print("Translation vector between cameras (T):")
print(T.T)  # transpose to show as row for readability


# -----------------------
#  RECTIFICATION
# -----------------------

print("\nComputing rectification transforms...")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtxL, distL,
    mtxR, distR,
    img_shape,
    R, T,
    alpha=0.0  # no black borders
)

left_map1, left_map2 = cv2.initUndistortRectifyMap(
    mtxL, distL, R1, P1, img_shape, cv2.CV_16SC2
)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    mtxR, distR, R2, P2, img_shape, cv2.CV_16SC2
)

# -----------------------
#  SAVE CALIBRATION
# -----------------------

np.savez(
    OUTPUT_CALIB_FILE,
    mtxL=mtxL, distL=distL,
    mtxR=mtxR, distR=distR,
    R=R, T=T, E=E, F=F, Q=Q,
    R1=R1, R2=R2, P1=P1, P2=P2,
    left_map1=left_map1, left_map2=left_map2,
    right_map1=right_map1, right_map2=right_map2
)

print(f"\nCalibration complete. Data saved to '{OUTPUT_CALIB_FILE}'.")
