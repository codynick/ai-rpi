import sys
import capture
import Olds.opencv_ocr as opencv_ocr

# capture.capture()

# Use command-line argument if provided, else use default
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "samples/hello.png"

print(opencv_ocr.opencv_ocr(filename))
