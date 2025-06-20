import cv2
import numpy as np
import glob
import os

# Checkerboard internal corners (not number of squares!)
CHECKERBOARD = (10, 7)
SQUARE_SIZE = 25.0  # millimeters

# Prepare object points (real world coordinates)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load all calibration images
images = glob.glob("calibration_images/*.jpg")  # adjust path if needed

if not images:
    print("❌ No images found in 'calibration_images/' folder.")
    exit()

print(f"✅ Found {len(images)} calibration images.")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"🔍 Processing {fname}...")

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        preview = cv2.resize(img, (960, 720))
        cv2.imshow('Calibration Preview', preview)
        cv2.waitKey(100)
    else:
        print(f"⚠️  Checkerboard not detected in {fname}")

cv2.destroyAllWindows()

# Check if enough valid views were collected
if len(objpoints) < 5:
    print("❌ Not enough valid images for calibration.")
    exit()

# Calibrate the camera
print("⚙️  Running calibration...")
try:
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    if not ret:
        print("❌ Calibration failed to converge.")
        exit()

    print("✅ Calibration successful!")
    print("Camera Matrix (K):\n", K)
    print("Distortion Coefficients (D):\n", D)

    # Save results
    np.savez("camera_calibration.npz", K=K, D=D)
    np.savetxt("K.txt", K)
    np.savetxt("D.txt", D)

    print("📁 Calibration files saved as 'camera_calibration.npz', 'K.txt', and 'D.txt'.")

except Exception as e:
    print("❌ An error occurred during calibration or saving:")
    print(str(e))