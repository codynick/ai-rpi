# ai-rpi
AI modules for CodyNick on Raspberry Pi 4
Awesome—since you’ve locked your “final” script, here’s a clean, reproducible setup you can run on a fresh **Ubuntu 24.04 LTS (aarch64)** box to get *all four tasks* (OCR / Objects / Emotion / Age via InsightFace) working.

---

# Stage 0 — Sanity check (arch + Python)

```bash
uname -m
python3 --version
# Expect:
# aarch64
# Python 3.12.x
```

---

# Stage 1 — System packages

```bash
sudo apt update
sudo apt install -y \
  python3.12 python3.12-venv python3.12-dev \
  build-essential pkg-config libopenblas-dev \
  libgl1 libglib2.0-0 \
  tesseract-ocr libtesseract-dev \
  git wget curl
```

---

# Stage 2 — Project & virtualenv

```bash
# Clone your repo (or copy your script folder)
git clone https://github.com/codynick/ai
cd ai

# venv
python3.12 -m venv venv
source venv/bin/activate

# Faster wheels on Raspberry Pi / ARM
pip install -U pip setuptools wheel
pip config set global.extra-index-url https://www.piwheels.org/simple
```

---

# Stage 3 — Python dependencies

## 3A) Core, CPU-friendly stack (OCR + YOLO + ONNX + OpenCV)

```bash
pip install --only-binary=:all: \
  numpy==1.26.4 \
  opencv-python-headless==4.9.0.80 \
  pillow==10.4.0 \
  pytesseract==0.3.13 \
  onnxruntime==1.23.2
```

## 3B) Age: **InsightFace** (CPU)

> InsightFace pulls its own model zoo and does age/gender with `FaceAnalysis` on CPU.

```bash
# Use PyPI as primary for this one to avoid missing wheels on piwheels
pip install --no-cache-dir --index-url https://pypi.org/simple \
  insightface==0.7.3
```

## 3C) Emotion backends

### Option 1 (default/recommended on Pi): **DeepFace + MTCNN**

```bash
pip install deepface==0.0.93 mtcnn
```

### Option 2 (no TF needed): **FER+ ONNX** path

*(Already satisfied by onnxruntime; no extra pip needed)*

### Option 3 (optional): **DeepFace + RetinaFace (TF path)**

If you want RetinaFace detector for emotion (sometimes better than MTCNN), add:

```bash
pip install tf-keras tensorflow==2.20.0
# DeepFace is already installed; RetinaFace will route via tf-keras
```

> If you see `ValueError: requires tf-keras`, just `pip install tf-keras` (you already pinned TF above).

---

# Stage 4 — Models directory & weights

```bash
mkdir -p models
```

**YOLOv8n (objects):**

```bash
wget -O models/yolov8n.onnx \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx
```

**OpenCV Face SSD (used by FER+ backend only):**

```bash
wget -O models/deploy.prototxt \
  https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/deploy.prototxt

wget -O models/res10_300x300_ssd_iter_140000.caffemodel \
  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
```

**FER+ ONNX (emotion backend “onnx”):**

```bash
wget -O models/emotion-ferplus.onnx \
  https://huggingface.co/webml/models-moved/resolve/main/emotion-ferplus-8.onnx
```

> (Age via InsightFace needs **no** extra downloads; models are fetched/cached on first run.)

---

# Stage 5 — Quick health check

```bash
python - <<'PY'
import sys, platform
import numpy, cv2, onnxruntime, PIL, pytesseract
print("Python:", sys.version.split()[0], "| Arch:", platform.machine())
print("numpy:", numpy.__version__, "| opencv:", cv2.__version__, "| ort:", onnxruntime.__version__)
import insightface
print("insightface:", insightface.__version__)
try:
    from deepface import DeepFace
    print("deepface: OK")
except Exception as e:
    print("deepface import error:", e)
PY
```

Expected (or close):

```
Python: 3.12.x | Arch: aarch64
numpy: 1.26.4 | opencv: 4.9.0 | ort: 1.23.2
insightface: 0.7.3
deepface: OK
```

---

# Stage 6 — Smoke tests (run from repo root)

```bash
# OCR
python unified_vision_tools_final_rpi7.py --task ocr --image samples/hello.png

# Objects
python unified_vision_tools_final_rpi7.py --task objects --image samples/objects.jpg \
  --model models/yolov8n.onnx --conf 0.50 --iou 0.50

# Age (InsightFace)
python unified_vision_tools_final_rpi7.py --task age --image samples/Happy.jpg

# Emotion (DeepFace + MTCNN default)
python unified_vision_tools_final_rpi7.py --task emotion --image samples/Happy.jpg

# Emotion with RetinaFace (requires TF path from Stage 3C) -- Parameter NOT passed now
python unified_vision_tools_final_rpi7.py --task emotion --image samples/Happy.jpg \
  --emotion-backend deepface-retinaface

# Emotion with FER+ ONNX -- Parameter NOT passed now
python unified_vision_tools_final_rpi7.py --task emotion --image samples/Happy.jpg \
  --emotion-backend onnx
```

Outputs are saved next to your input image as:

* `*_ocr.jpg/.json`
* `*_objects.jpg/.json`
* `*_emotion.jpg/.json`
* `*_age.jpg/.json`

---

# Stage 7 — Common fixes

* **DeepFace shows “neutral” a lot**
  Try: better-lit portrait images (frontal face), or switch backends:
  `--emotion-backend deepface-retinaface` (after Stage 3C)
  or `--emotion-backend onnx` (FER+).

* **`ValueError: requires tf-keras`**
  `pip install tf-keras` (and ensure `tensorflow==2.20.0` is installed).

* **ONNXRuntime provider errors**
  Ensure you’re using the CPU provider only (the script does), and that `onnxruntime==1.23.2` installed cleanly.

* **InsightFace slow on first run**
  First call downloads models to cache (`~/.insightface`); subsequent runs are much faster.

---

This setup matches the capabilities and backends wired into your final script (tasks, flags, model paths, and JSON/image outputs). 
