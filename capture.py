import cv2
import time
import os
import platform

def capture():
    # Create output folder
    os.makedirs("captures", exist_ok=True)

    # Detect OS and set camera device
    if platform.system() == "Linux":
        device = "/dev/video0"
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    elif platform.system() == "Windows":
        device = 0  # Default webcam index
        cap = cv2.VideoCapture(device)
    else:
        raise RuntimeError("Unsupported OS")

    # Set resolution and format (MJPG works on most devices)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {device}")

    print("Warming up camera...")

    # Warm up camera — discard first few frames
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    # Try capturing a valid frame
    ret, frame = cap.read()
    tries = 0
    while (not ret or frame is None or frame.size == 0) and tries < 10:
        time.sleep(0.2)
        ret, frame = cap.read()
        tries += 1

    if not ret or frame is None or frame.size == 0:
        raise RuntimeError("Could not capture a valid frame (camera may need replug or permission fix)")
        return
    
    filename = f"captures/photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)
    cap.release()

    print(f"Photo saved to {filename} from {device}")

    return filename
