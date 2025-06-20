import cv2
import platform
import numpy as np

# Adjust this if you have more than 10 possible devices
MAX_CAMERAS = 10

def get_backend():
    """Use DirectShow on Windows, default elsewhere."""
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY

def find_cameras(max_idx=MAX_CAMERAS):
    """Return a list of camera indices that open and return at least one frame."""
    backend = get_backend()
    found = []
    for idx in range(max_idx):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            continue
        ret, _ = cap.read()
        cap.release()
        if ret:
            found.append(idx)
    return found

def main():
    cams = find_cameras()
    if not cams:
        print("No cameras found.")
        return

    print(f"Found cameras: {cams}")
    backend = get_backend()
    # Open VideoCapture objects for each found camera
    caps = [cv2.VideoCapture(idx, backend) for idx in cams]
    # (Optional) set all to 640×480
    for cap in caps:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            for cap, idx in zip(caps, cams):
                ret, frame = cap.read()
                if ret:
                    cv2.imshow(f"Camera {idx}", frame)
                else:
                    # show a blank image if this one fails
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.imshow(f"Camera {idx}", blank)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
