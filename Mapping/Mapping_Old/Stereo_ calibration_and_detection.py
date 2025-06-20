import cv2
import numpy as np
import os
from tqdm import tqdm
import time

# Configuration
CHECKERBOARD = (6, 8)
SQUARE_SIZE = 25.0  # mm
calibration_file = "stereo_calib_data.npz"

cam_left = 0
cam_right = 1
frame_width, frame_height = 1280, 720
display_size = (640, 480)

lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])


from tqdm import tqdm

import time

def run_calibration():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints, imgpoints_left, imgpoints_right = [], [], []

    capL = cv2.VideoCapture(cam_left)
    capR = cv2.VideoCapture(cam_right)
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    print("Press 'w' to finish and calibrate. Press 'q' to quit.")

    last_capture_time = time.time()

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
        foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

        viewL = frameL.copy()
        viewR = frameR.copy()

        current_time = time.time()
        if foundL and foundR and len(objpoints) < 30 and (current_time - last_capture_time) >= 0.5:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            objpoints.append(objp.copy())
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)

            last_capture_time = current_time
            print(f"Collected {len(objpoints)} valid samples")

        elif len(objpoints) >= 30:
            print("Reached 30 samples. Press 'w' to calibrate.")

        preview = np.hstack((
            cv2.resize(viewL, display_size),
            cv2.resize(viewR, display_size)
        ))
        cv2.imshow("Stereo Calibration", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            break
        elif key == ord('q'):
            capL.release()
            capR.release()
            cv2.destroyAllWindows()
            exit()

    if len(objpoints) < 10:
        print(f"Only {len(objpoints)} samples collected. At least 10 recommended.")
        capL.release()
        capR.release()
        cv2.destroyAllWindows()
        exit()

    image_size = (frameL.shape[1], frameL.shape[0])
    print("Calibration image size:", image_size)

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    _, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, image_size, None, None)
    _, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, image_size, None, None)

    flags = cv2.CALIB_FIX_INTRINSIC
    _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K1, D1, K2, D2, image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        flags=flags
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T)

    np.savez(calibration_file,
             K1=K1, D1=D1, K2=K2, D2=D2,
             R=R, T=T, Q=Q, P1=P1, P2=P2, R1=R1, R2=R2,
             image_size=image_size)

    print("Calibration complete.")
    print("Baseline:", np.linalg.norm(T))

    return K1, D1, K2, D2, Q, P1, P2, R1, R2, image_size

def load_calibration():
    if not os.path.exists(calibration_file):
        print("Calibration file not found.")
        exit()
    data = np.load(calibration_file, allow_pickle=True)
    return (
        data['K1'], data['D1'], data['K2'], data['D2'],
        data['Q'], data['P1'], data['P2'], data['R1'], data['R2'],
        tuple(data['image_size'])
    )

def detect_and_measure(K1, D1, K2, D2, Q, P1, P2, R1, R2, calibration_image_size):
    capL = cv2.VideoCapture(cam_left)
    capR = cv2.VideoCapture(cam_right)
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not capL.isOpened() or not capR.isOpened():
        print("Failed to open one or both cameras.")
        return

    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Could not grab initial frames.")
        return

    detection_image_size = (frameL.shape[1], frameL.shape[0])
    print("Calibration image size:", calibration_image_size)
    print("Detection image size:", detection_image_size)

    if detection_image_size != calibration_image_size:
        print("Mismatch between calibration and detection resolutions.")
        print("Please recalibrate at this resolution.")
        capL.release()
        capR.release()
        return

    mapL1, mapL2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, detection_image_size, cv2.CV_16SC2)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, detection_image_size, cv2.CV_16SC2)

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

    print("Running detection. Press 'q' to quit.")
    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("Failed to read frames.")
            break

        rectL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        hsv = cv2.cvtColor(rectL, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                if 0 <= cx < detection_image_size[0] and 0 <= cy < detection_image_size[1]:
                    world_point = points_3D[cy, cx]
                    if not np.isinf(world_point[2]) and world_point[2] > 0:
                        depth_cm = world_point[2] * 100
                        width_cm = (w * world_point[2]) / P1[0, 0] * 100
                        height_cm = (h * world_point[2]) / P1[1, 1] * 100

                        cv2.rectangle(rectL, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(rectL, f"D: {depth_cm:.1f} cm", (x, y - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(rectL, f"W: {width_cm:.1f} cm H: {height_cm:.1f} cm", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        combined = np.hstack((
            cv2.resize(rectL, display_size),
            cv2.resize(rectR, display_size)
        ))
        cv2.imshow("Stereo | Blue Block Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Select mode:")
    print("1. Run stereo calibration (automatically starts detection)")
    print("2. Use existing calibration and start detection")

    choice = input("Enter 1 or 2: ")

    if choice == "1":
        K1, D1, K2, D2, Q, P1, P2, R1, R2, calibration_image_size = run_calibration()
        detect_and_measure(K1, D1, K2, D2, Q, P1, P2, R1, R2, calibration_image_size)

    elif choice == "2":
        if not os.path.exists(calibration_file):
            print("❌ No calibration file found. Please run calibration first.")
            exit()
        K1, D1, K2, D2, Q, P1, P2, R1, R2, calibration_image_size = load_calibration()
        detect_and_measure(K1, D1, K2, D2, Q, P1, P2, R1, R2, calibration_image_size)

    else:
        print("❌ Invalid choice.")
        exit()