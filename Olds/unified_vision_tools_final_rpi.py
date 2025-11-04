#!/usr/bin/env python3
"""
Unified Vision Tool — Final

Tasks:
  • OCR (Tesseract via pytesseract)
  • Emotion (FER+ ONNX; optional face align with InsightFace)
  • Objects (YOLO ONNX via onnxruntime)
  • Age  (choose backend: onnx=InsightFace, opencv=Caffe, deepface=TF)

Examples:
  python unified_vision_tools_final.py --task ocr --image samples/hello.png
  python unified_vision_tools_final.py --task objects --image samples/objects.jpg --model models/yolov8n.onnx --conf 0.6 --iou 0.5
  python unified_vision_tools_final.py --task emotion --image samples/Happy.jpg
  python unified_vision_tools_final.py --task age --image samples/Happy.jpg --age-backend onnx

Notes:
  • Age default backend is ONNX (InsightFace) for best portability (no TF required).
  • Emotion runs FER+ ONNX (8 classes). If InsightFace is installed, it will align the face.
  • Place models in ./models (see URLs in code). The tool will error clearly if a file is missing.
"""

from __future__ import annotations
import argparse
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

# ==== Model URLs (for reference) ====
FERPLUS_ONNX_URL = (
    "https://huggingface.co/webml/models-moved/resolve/main/emotion-ferplus-8.onnx"
)
YOLOV8N_ONNX_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
)
FACE_PROTO_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/deploy.prototxt"
)
FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/"
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)
AGE_PROTO_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
)
AGE_MODEL_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_net.caffemodel"
)

# ==== Paths ====
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)
FERPLUS_ONNX = MODELS / "emotion-ferplus.onnx"
YOLO_ONNX = MODELS / "yolov8n.onnx"
FACE_PROTO = MODELS / "deploy.prototxt"
FACE_CAFFE = MODELS / "res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO = MODELS / "age_deploy.prototxt"
AGE_CAFFE = MODELS / "age_net.caffemodel"


# ---------- Utilities ----------
def ensure_present(path: Path, hint: str) -> None:
    """Raise a clear error if a required model file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required model file: {path}\nHint: {hint}"
        )


def save_outputs(img_bgr: np.ndarray, img_path: Path, suffix: str, dets: dict | None = None) -> None:
    out_img = img_path.with_stem(img_path.stem + f"_{suffix}").with_suffix(".jpg")
    out_json = img_path.with_stem(img_path.stem + f"_{suffix}").with_suffix(".json")
    cv2.imwrite(str(out_img), img_bgr)
    if dets is not None:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(dets, f, ensure_ascii=False, indent=2)
    print(f"Saved image: {out_img}")
    if dets is not None:
        print(f"Saved JSON:  {out_json}")


# ---------- 1) OCR ----------
def run_ocr(image_path: str) -> None:
    import pytesseract

    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    elif platform.system() == "Linux":
        pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6", lang="eng").strip()

    annotated = img.copy()
    snippet = (text[:80] + "…") if len(text) > 80 else text
    if snippet:
        cv2.rectangle(annotated, (5, 5), (min(700, 15 * len(snippet)), 35), (255, 255, 255), -1)
        cv2.putText(annotated, snippet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    save_outputs(annotated, Path(image_path), "ocr", {"image": image_path, "text": text})
    print("\n--- OCR RESULT ---\n" + text + "\n------------------\n")


# ---------- 2) Emotion (FER+ ONNX) ----------
def _align_face_with_insightface(img_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int,int,int,int] | None]:
    """Optional 5pt alignment via InsightFace if available; returns (roi64, used_box)."""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(img_bgr)
        if not faces:
            return None, None
        H, W = img_bgr.shape[:2]
        f = max(faces, key=lambda z: (z.bbox[2]-z.bbox[0])*(z.bbox[3]-z.bbox[1]))
        x1, y1, x2, y2 = map(int, f.bbox)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        s = 1.05 * max(x2 - x1, y2 - y1)
        x1, y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        x2, y2 = int(min(W - 1, cx + s / 2)), int(min(H - 1, cy + s / 2))
        used_box = (x1, y1, x2, y2)

        kps = getattr(f, "kps", None)
        if kps is not None and np.shape(kps) == (5, 2):
            src = kps.astype(np.float32)
            ref = np.array(
                [[38.2946, 51.6963],
                 [73.5318, 51.5014],
                 [56.0252, 71.7366],
                 [41.5493, 92.3655],
                 [70.7299, 92.2041]], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
            if M is not None:
                aligned112 = cv2.warpAffine(img_bgr, M, (112, 112), flags=cv2.INTER_LINEAR)
                c0 = 8
                crop96 = aligned112[c0:112-c0, c0:112-c0]
                roi = cv2.resize(crop96, (64, 64), interpolation=cv2.INTER_AREA)
                return roi, used_box

        # fallback to centered square crop around bbox
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size:
            h, w = crop.shape[:2]
            m = max(h, w)
            pad = np.zeros((m, m, 3), crop.dtype)
            yoff, xoff = (m - h) // 2, (m - w) // 2
            pad[yoff:yoff+h, xoff:xoff+w] = crop
            roi = cv2.resize(pad, (64, 64), interpolation=cv2.INTER_AREA)
            return roi, used_box
    except Exception:
        pass
    return None, None


def run_emotion(image_path: str) -> None:
    import onnxruntime as ort

    labels = [
        "neutral","happiness","surprise","sadness",
        "anger","disgust","fear","contempt"
    ]
    ensure_present(FERPLUS_ONNX, f"Download FER+ ONNX to {FERPLUS_ONNX} (e.g.\n  wget -O {FERPLUS_ONNX} \"{FERPLUS_ONNX_URL}\" )")

    img0 = cv2.imread(image_path)
    if img0 is None:
        raise FileNotFoundError(image_path)

    roi, used_box = _align_face_with_insightface(img0)
    if roi is None:
        # simple whole-image fallback
        roi = cv2.resize(img0, (64, 64), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    x = gray.astype(np.float32) / 255.0
    x = x[np.newaxis, np.newaxis, :, :]  # [1,1,64,64]

    sess = ort.InferenceSession(str(FERPLUS_ONNX), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    logits = sess.run(None, {inp_name: x})[0]  # [1,8]

    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    probs = probs.flatten()
    idx = int(probs.argmax())
    label, conf = labels[idx], float(probs[idx])

    annotated = img0.copy()
    cv2.putText(annotated, f"{label} {conf:.2f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    if used_box is not None:
        x1, y1, x2, y2 = used_box
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)

    save_outputs(
        annotated, Path(image_path), "emotion",
        {"image": image_path, "emotion": label, "confidence": conf,
         "probs": {l: float(p) for l, p in zip(labels, probs)}}
    )
    print(f"Emotion: {label} ({conf:.2f})")


# ---------- 3) Objects (YOLO ONNX) ----------
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]


def _letterbox(im: np.ndarray, new_shape=640):
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    im = cv2.resize(im, (nw, nh))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (left, top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_th=0.5):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (
            (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            + (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
            - inter + 1e-6
        )
        idxs = idxs[1:][iou <= iou_th]
    return keep


def run_objects(image_path: str, conf_th=0.35, iou_th=0.5, model_override: str | None = None, debug: bool = False) -> None:
    import onnxruntime as ort

    model_path = Path(model_override) if model_override else YOLO_ONNX
    ensure_present(model_path, f"Download YOLO ONNX to {model_path} (e.g.\n  wget -O {model_path} \"{YOLOV8N_ONNX_URL}\" )")

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    img0 = cv2.imread(image_path)
    if img0 is None:
        raise FileNotFoundError(image_path)

    inp, r, (dx, dy) = _letterbox(img0.copy(), 640)
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    blob = (inp.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    out = sess.run(None, {inp_name: blob})[0]
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    dets = []
    annotated = img0.copy()

    if out.ndim == 2 and (out.shape[1] in (84, 85) or out.shape[0] in (84, 85)):
        if out.shape[0] in (84, 85):
            out = out.T
        N, D = out.shape
        xywh = out[:, :4].astype(np.float32).copy()
        if D == 84:
            cls = out[:, 4:].astype(np.float32).copy()
            if cls.max() > 1.0 or cls.min() < 0.0:
                cls = 1.0 / (1.0 + np.exp(-cls))
            scores = cls
        elif D == 85:
            obj = out[:, 4:5].astype(np.float32).copy()
            cls = out[:, 5:].astype(np.float32).copy()
            if obj.max() > 1.0 or obj.min() < 0.0:
                obj = 1.0 / (1.0 + np.exp(-obj))
            if cls.max() > 1.0 or cls.min() < 0.0:
                cls = 1.0 / (1.0 + np.exp(-cls))
            scores = obj * cls
        else:
            raise RuntimeError(f"Unexpected YOLO D={D}")

        conf = scores.max(1)
        class_ids = scores.argmax(1)
        if xywh.max() <= 2.0:
            xywh *= 640.0
        xyxy = np.empty_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        m = conf > conf_th
        if not m.any():
            m = conf > max(0.10, conf_th * 0.5)
        xyxy, conf, class_ids = xyxy[m], conf[m], class_ids[m]
        keep = _nms(xyxy, conf, iou_th) if len(xyxy) else []
        xyxy, conf, class_ids = xyxy[keep], conf[keep], class_ids[keep]

    elif out.ndim == 2 and out.shape[1] == 6:
        xyxy = out[:, :4].astype(np.float32).copy()
        conf = out[:, 4].astype(np.float32).copy()
        class_ids = out[:, 5].astype(np.int32).copy()
        m = conf > conf_th
        xyxy, conf, class_ids = xyxy[m], conf[m], class_ids[m]

    elif out.ndim == 1 and out.shape[0] == 6:
        xyxy = out[:4][None, :].astype(np.float32)
        conf = np.array([float(out[4])], dtype=np.float32)
        class_ids = np.array([int(out[5])], dtype=np.int32)

    else:
        raise RuntimeError(f"Unhandled ONNX output layout: {out.shape}")

    if len(xyxy):
        xyxy[:, [0, 2]] -= dx
        xyxy[:, [1, 3]] -= dy
        xyxy /= r
        xyxy = xyxy.clip(
            [0, 0, 0, 0],
            [img0.shape[1]-1, img0.shape[0]-1, img0.shape[1]-1, img0.shape[0]-1]
        )

    for (x1, y1, x2, y2), c, cls in zip(xyxy, conf, class_ids):
        name = COCO80[int(cls)] if 0 <= int(cls) < len(COCO80) else str(int(cls))
        dets.append({
            "class_name": name,
            "confidence": float(c),
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        })
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(annotated, f"{name} {c:.2f}", (int(x1), max(0, int(y1)-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if len(dets):
        print("\nDetected objects:")
        for d in dets:
            print(f"  • {d['class_name']} ({d['confidence']:.2f})")
    else:
        print("\nNo objects detected.")

    save_outputs(annotated, Path(image_path), "objects",
                 {"image": image_path, "num_detections": len(dets), "detections": dets})


# ---------- 4) Age (3 backends) ----------
AGE_BUCKETS = ['(0-2)','(4-6)','(8-12)','(15-20)','(21-24)','(25-32)','(38-43)','(48-53)','(60-100)']


def run_age_opencv(image_path: str) -> None:
    # Sanity: all four files must exist
    for pth, name, url in [
        (FACE_PROTO, "face prototxt", FACE_PROTO_URL),
        (FACE_CAFFE, "face model", FACE_MODEL_URL),
        (AGE_PROTO,  "age prototxt", AGE_PROTO_URL),
        (AGE_CAFFE,  "age model", AGE_MODEL_URL),
    ]:
        ensure_present(pth, f"Download with:\n  wget -O {pth} \"{url}\"")

    face_net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_CAFFE))
    age_net  = cv2.dnn.readNetFromCaffe(str(AGE_PROTO),  str(AGE_CAFFE))

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    dets = face_net.forward()

    annotated = img.copy()
    results = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = (dets[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1, x2 = np.clip([x1, x2], 0, w-1)
        y1, y2 = np.clip([y1, y2], 0, h-1)
        if x2 <= x1 or y2 <= y1:
            continue
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue
        fblob = cv2.dnn.blobFromImage(cv2.resize(face, (227, 227)), 1.0, (227, 227),
                                      (78.4263, 87.7689, 114.8958), swapRB=False)
        age_net.setInput(fblob)
        pred = age_net.forward()[0]
        age_id = int(np.argmax(pred))
        age_conf = float(np.max(pred))
        age_label = AGE_BUCKETS[age_id] if 0 <= age_id < len(AGE_BUCKETS) else str(age_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(annotated, f"{age_label} {age_conf:.2f}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        results.append({"bbox_xyxy": [x1, y1, x2, y2], "age_bucket": age_label, "confidence": age_conf})

    save_outputs(annotated, Path(image_path), "age", {"image": image_path, "detections": results})


def run_age_deepface(image_path: str) -> None:
    from deepface import DeepFace

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    res = DeepFace.analyze(img_path=img, actions=['age'], detector_backend='retinaface', enforce_detection=False)
    if isinstance(res, dict):
        res = [res]

    annotated = img.copy()
    dets = []
    for r in res or []:
        region = r.get('region') or {}
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
        age_value = float(r.get('age', -1))
        if w > 0 and h > 0:
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(annotated, f"age~{int(round(age_value))}", (x, max(0, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        def to_bucket(a: float) -> str:
            if a < 3: return "(0-2)"
            if a < 7: return "(4-6)"
            if a < 13: return "(8-12)"
            if a < 21: return "(15-20)"
            if a < 25: return "(21-24)"
            if a < 33: return "(25-32)"
            if a < 44: return "(38-43)"
            if a < 54: return "(48-53)"
            return "(60-100)"
        dets.append({
            "bbox_xyxy": [int(x), int(y), int(x+w), int(y+h)],
            "age_estimate": age_value,
            "age_bucket": to_bucket(age_value),
        })

    save_outputs(annotated, Path(image_path), "age", {"image": image_path, "detections": dets})


def run_age_insightface(image_path: str) -> None:
    import insightface
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    faces = app.get(img)
    annotated = img.copy()
    dets = []
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        age = int(getattr(f, 'age', -1))
        gender = int(getattr(f, 'gender', -1))  # 0=female, 1=male
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"age~{age}" if age >= 0 else "age~?"
        if gender in (0, 1):
            label += " F" if gender == 0 else " M"
        cv2.putText(annotated, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        dets.append({
            "bbox_xyxy": [x1, y1, x2, y2],
            "age_estimate": age if age >= 0 else None,
            "gender": {0: "female", 1: "male"}.get(gender),
        })

    save_outputs(annotated, Path(image_path), "age", {"image": image_path, "detections": dets})


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["ocr", "emotion", "objects", "age"])
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--model", default=None, help="[objects] path to YOLO ONNX file (default models/yolov8n.onnx)")
    ap.add_argument("--debug", action="store_true", help="[objects] print confidence stats")
    ap.add_argument("--age-backend", choices=["onnx", "opencv", "deepface"], default="onnx",
                    help="Age backend: onnx (InsightFace) | opencv (Caffe) | deepface (TF)")

    args = ap.parse_args()
    start = time.time()

    if args.task == "ocr":
        run_ocr(args.image)

    elif args.task == "emotion":
        run_emotion(args.image)

    elif args.task == "objects":
        run_objects(args.image, conf_th=args.conf, iou_th=args.iou, model_override=args.model, debug=args.debug)

    elif args.task == "age":
        if args.age_backend == "onnx":
            run_age_insightface(args.image)
        elif args.age_backend == "opencv":
            run_age_opencv(args.image)
        else:
            # Try DeepFace; if it fails, fall back to InsightFace
            try:
                run_age_deepface(args.image)
            except Exception as e:
                print(f"[deepface] failed ({e}); falling back to InsightFace ONNX…")
                run_age_insightface(args.image)

    print(f"Total elapsed time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
