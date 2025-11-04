import cv2
import time
import os
import platform
import pytesseract
import numpy as np

def opencv_ocr(filename):
    # --- Detect OS and set Tesseract path ---
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    elif platform.system() == "Linux":
        pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    else:
        raise RuntimeError("Unsupported OS for Tesseract OCR")

    # --- Image file path ---
    # filename = os.path.join("captures", "hello.png")
    # filename = "captures/hello.png"  # Example filename; replace with actual captured file

    # --- Read image ---
    time1 = time.time()
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise RuntimeError("Could not read the saved image for OCR.")
    print("Image Read Time:", round((time.time() - time1) * 10) / 10)

    # --- Convert to grayscale ---
    time1 = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("cv2.COLOR_BGR2GRAY Time:", round((time.time() - time1) * 10) / 10)

    # --- OCR configuration ---
    ocr_config = r'--oem 3 --psm 6'

    # --- Run OCR ---
    time1 = time.time()
    text = pytesseract.image_to_string(gray, config=ocr_config, lang='eng')

    # --- Output ---
    print("\n--- OCR RESULT ---\n")
    print(text.strip() if text else "")
    print("\n------------------\n")
    print("OCR Time:", round((time.time() - time1) * 10) / 10)

    return(text.strip() if text else "")