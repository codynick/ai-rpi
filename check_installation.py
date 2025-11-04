import cv2, numpy, onnxruntime, pytesseract
from fer.fer import FER
print("✅ OpenCV:", cv2.__version__)
print("✅ NumPy:", numpy.__version__)
print("✅ ONNXRuntime:", onnxruntime.__version__)
print("✅ FER OK:", FER)
print("✅ Tesseract path:", pytesseract.pytesseract.tesseract_cmd)