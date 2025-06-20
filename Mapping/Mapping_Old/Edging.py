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
    Create a StereoSGBM matcher with given parameters (unused here but kept for completeness).
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

def main():
    # 1) Load stereo rectification maps (replace 'stereo_calib_live.npz' if your file differs)
    left_map1, left_map2, right_map1, right_map2, Q = load_calibration('stereo_calib_live.npz')

    # 2) (Optional) Create stereo matcher if you want disparity later
    # stereo = create_stereo_matcher()

    # 3) Open camera streams (adjust indices if needed)
    capL = cv2.VideoCapture(0)
    capR = cv2.VideoCapture(1)
    if not capL.isOpened() or not capR.isOpened():
        print("ERROR: Could not open one or both cameras.")
        return

    # 4) Create named windows to display outputs
    cv2.namedWindow('Left Camera Raw')
    cv2.namedWindow('Right Camera Raw')
    cv2.namedWindow('Rectified Left')
    cv2.namedWindow('Edges (Canny)')
    cv2.namedWindow('Contours Overlay')

    print("Press 'q' to quit.")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("ERROR: Could not read from cameras.")
            break

        # 5) Show raw camera feeds
        cv2.imshow('Left Camera Raw', frameL)
        cv2.imshow('Right Camera Raw', frameR)

        # 6) Rectify the left frame
        rectL = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)

        # 7) Convert to grayscale & blur to reduce noise
        gray = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 8) Run Canny to detect edges (tweak thresholds 50, 150 as needed)
        edges = cv2.Canny(blurred, 50, 150)
        cv2.imshow('Edges (Canny)', edges)

        # 9) Find contours on the binary edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 10) Draw all contours on a copy of rectified left
        contour_overlay = rectL.copy()
        # Draw each contour in green, thickness=2
        cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Contours Overlay', contour_overlay)

        # 11) Also show the plain rectified left
        cv2.imshow('Rectified Left', rectL)

        # 12) Break if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 13) Cleanup
    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
