import time
start = time.time()

import cv2
from fer.fer import FER
import sys
# import warnings
# warnings.filterwarnings("ignore")

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

print(f"{RED}Import time: {round((time.time() - start) * 10) / 10} seconds{RESET}")
start = time.time()

# Get filename from command-line argument or use default
filename = sys.argv[1] if len(sys.argv) > 1 else "samples/Happy.jpg"

# Load image
image = cv2.imread(filename)
if image is None:
    raise FileNotFoundError(f"Image not found: {filename}")

print(f"{RED}Image Load time: {round((time.time() - start) * 10) / 10} seconds{RESET}")
start = time.time()

# Convert to RGB and detect emotions
detector = FER(mtcnn=True)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = detector.detect_emotions(rgb)

print(f"{RED}Detection time: {round((time.time() - start) * 10) / 10} seconds{RESET}")

print(results)

# Print results
if not results:
    print("No face detected.")
else:
    for i, r in enumerate(results, 1):
        emotions = r["emotions"]
        emotions_filtered = {k: v for k, v in emotions.items() if k.lower() != "neutral"}
        if not emotions_filtered:
            print(f"{GREEN}Person {i}: All neutral.{RESET}")
            continue
        dominant = max(emotions_filtered, key=emotions_filtered.get)
        confidence = emotions_filtered[dominant] * 100
        print(f"{GREEN}Person {i}: {YELLOW}{dominant} ({confidence:.1f}%){RESET}")