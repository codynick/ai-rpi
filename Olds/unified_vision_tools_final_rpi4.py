#!/usr/bin/env python3
"""
unified_vision_tools_final_rpi4.py

Tools:
  1) OCR                  -> pytesseract (Tesseract OCR)
  2) Emotion detection    -> DeepFace (detector_backend='mtcnn', Keras/TF + tf-keras)
  3) Object detection     -> YOLO ONNX (onnxruntime)
  4) Human age detection  -> OpenCV DNN (Caffe SSD face + age_net)

Outputs:
  Annotated "<image_stem>_<task>.jpg" and JSON "<image_stem>_<task>.json"
"""

import argparse
import json
import os
import platform
from pathlib import Path
import time

import cv2
import numpy as np

# ---------------- Utilities ----------------
def save_outputs(img_bgr, img_path: Path, suffix: str, dets=None):
    out_img = img_path.with_stem(img_path.stem + f"_{suffix}").with_suffix(".jpg")
    out_json = img_path.with_stem(img_path.stem + f"_{suffix}").with_suffix(".json")
    cv2.imwrite(str(out_img), img_bgr)
    if dets is not None:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(dets, f, ensure_ascii=False, indent=2)
    print(f"Saved image: {out_img}")
    if dets is not None:
        print(f"Saved JSON:  {out_json}")


# ---------------- 1) OCR ----------------
def run_ocr(image_path: str):
    import pytesseract

    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
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


# ---------------- 2) Emotion (DeepFace + MTCNN) ----------------
def run_emotion_deepface_mtcnn(image_path: str):
    """
    Emotion analysis with DeepFace using MTCNN detector backend.
    Requires: tensorflow, keras (>=3.10), tf-keras, deepface, mtcnn
    """
    # Reduce TF logs / odds of OneDNN msgs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    try:
        from deepface import DeepFace
    except Exception as e:
        raise RuntimeError(
            "DeepFace import failed. Install deps:\n"
            "  pip install 'tensorflow==2.20.0' 'keras>=3.10.0' tf-keras deepface mtcnn\n"
        ) from e

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    # Run analysis
    res = DeepFace.analyze(
        img_path=img,
        actions=['emotion'],
        detector_backend='mtcnn',
        enforce_detection=False
    )

    # Normalize result to list
    if isinstance(res, dict):
        res = [res]

    # Annotate
    annotated = img.copy()
    dets = []
    for r in res or []:
        region = r.get('region') or {}
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
        emo_dict = r.get('emotion') or {}
        # Choose dominant (ignoring 'neutral' to avoid bias)
        emo_no_neutral = {k: v for k, v in emo_dict.items() if k.lower() != "neutral"}
        if emo_no_neutral:
            dominant = max(emo_no_neutral, key=emo_no_neutral.get)
            conf = float(emo_no_neutral[dominant])
        else:
            dominant = r.get('dominant_emotion', 'neutral')
            conf = float(emo_dict.get(dominant, 0.0) if isinstance(emo_dict, dict) else 0.0)

        if w > 0 and h > 0:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, f"{dominant} {conf:.1f}%", (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        dets.append({
            "bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
            "dominant_emotion": dominant,
            "confidence_percent": conf,
            "all_emotions": {k: float(v) for k, v in (emo_dict.items() if isinstance(emo_dict, dict) else [])}
        })

    # Quick console summary
    if dets:
        print("\nDetected emotions:")
        for d in dets:
            print(f"  • {d['dominant_emotion']} ({d['confidence_percent']:.1f}%) @ {d['bbox_xyxy']}")
    else:
        print("\nNo faces/emotions detected.")

    save_outputs(annotated, Path(image_path), "emotion", {
        "image": image_path,
        "num_faces": len(dets),
        "detections": dets
    })


# ---------------- 3) Objects (YOLO ONNX) ----------------
COCO80 = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
          "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
          "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
          "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
          "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
          "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
          "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
          "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
          "hair drier","toothbrush"]

def letterbox(im, new_shape=640):
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    im = cv2.resize(im, (nw, nh))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (left, top)

def nms(boxes, scores, iou_th=0.5):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]; keep.append(i)
        if idxs.size == 1: break
        xx1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[idxs[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[idxs[1:],3])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / ((boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
                       + (boxes[idxs[1:],2]-boxes[idxs[1:],0])*(boxes[idxs[1:],3]-boxes[idxs[1:],1])
                       - inter + 1e-6)
        idxs = idxs[1:][iou <= iou_th]
    return keep

def run_objects(image_path: str, conf_th=0.35, iou_th=0.5, model_override: str = None, debug: bool = False):
    import onnxruntime as ort

    model_path = Path(model_override) if model_override else Path("models/yolov8n.onnx")
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    img0 = cv2.imread(image_path)
    if img0 is None:
        raise FileNotFoundError(image_path)

    # Preprocess
    inp, r, (dx, dy) = letterbox(img0.copy(), 640)
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    blob = (inp.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # [1,3,640,640]

    # Inference
    outs = sess.run(None, {inp_name: blob})
    out = outs[0]
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    dets = []
    annotated = img0.copy()

    # Handle popular Ultralytics layouts
    if out.ndim == 2 and (out.shape[1] in (84, 85) or out.shape[0] in (84, 85)):
        if out.shape[0] in (84, 85):
            out = out.T  # -> [N,84/85]
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
            raise RuntimeError(f"Unexpected D: {D}")

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
        if not np.any(m):
            m = conf > max(0.10, conf_th * 0.5)
        xyxy, conf, class_ids = xyxy[m], conf[m], class_ids[m]
        keep = nms(xyxy, conf, iou_th) if len(xyxy) else []
        xyxy, conf, class_ids = xyxy[keep], conf[keep], class_ids[keep]

    elif out.ndim == 2 and out.shape[1] == 6:
        xyxy  = out[:, :4].astype(np.float32).copy()
        conf  = out[:, 4].astype(np.float32).copy()
        class_ids = out[:, 5].astype(np.int32).copy()
        m = conf > conf_th
        xyxy, conf, class_ids = xyxy[m], conf[m], class_ids[m]

    elif out.ndim == 1 and out.shape[0] == 6:
        xyxy  = out[:4][None, :].astype(np.float32)
        conf  = np.array([float(out[4])], dtype=np.float32)
        class_ids = np.array([int(out[5])], dtype=np.int32)

    else:
        raise RuntimeError(f"Unhandled ONNX output layout: {out.shape}")

    if len(xyxy):
        xyxy[:, [0, 2]] -= dx
        xyxy[:, [1, 3]] -= dy
        xyxy /= r
        xyxy = xyxy.clip([0, 0, 0, 0],
                         [img0.shape[1]-1, img0.shape[0]-1, img0.shape[1]-1, img0.shape[0]-1])

    for (x1, y1, x2, y2), c, cls in zip(xyxy, conf, class_ids):
        name = COCO80[int(cls)] if 0 <= int(cls) < len(COCO80) else str(int(cls))
        dets.append({"class_name": name, "confidence": float(c),
                     "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)]})
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


# ---------------- 4) Age (OpenCV DNN, Caffe) ----------------
def run_age(image_path: str):
    base_dir   = Path(__file__).resolve().parent
    models_dir = (base_dir / "models").resolve()

    age_p  = models_dir / "age_deploy.prototxt"
    age_m  = models_dir / "age_net.caffemodel"
    face_p = models_dir / "deploy.prototxt"
    face_m = models_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    for pth, name in [(age_p,"age prototxt"), (age_m,"age model"),
                      (face_p,"face prototxt"), (face_m,"face model")]:
        if not pth.exists():
            raise FileNotFoundError(f"Missing {name}: {pth}")

    print(f"[age] using face_p={face_p.name}, face_m={face_m.name}")
    print(f"[age] using age_p={age_p.name},  age_m={age_m.name}")

    # Preflight
    face_net = cv2.dnn.readNetFromCaffe(str(face_p), str(face_m))
    z = np.zeros((300,300,3), np.uint8)
    face_blob0 = cv2.dnn.blobFromImage(z, 1.0, (300,300), (104.0,177.0,123.0))
    face_net.setInput(face_blob0)
    _ = face_net.forward()

    age_net = cv2.dnn.readNetFromCaffe(str(age_p), str(age_m))
    z = np.zeros((227,227,3), np.uint8)
    age_blob0 = cv2.dnn.blobFromImage(z, 1.0, (227,227), (78.4263,87.7689,114.8958), swapRB=False)
    age_net.setInput(age_blob0)
    _ = age_net.forward()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]

    face_blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300),
                                      (104.0,177.0,123.0))
    face_net.setInput(face_blob)
    dets = face_net.forward()

    annotated = img.copy()
    results = []
    AGE_BUCKETS = ['(0-2)','(4-6)','(8-12)','(15-20)','(21-24)','(25-32)','(38-43)','(48-53)','(60-100)']

    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = (dets[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
        if x2 <= x1 or y2 <= y1:
            continue

        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_blob = cv2.dnn.blobFromImage(cv2.resize(face, (227,227)), 1.0, (227,227),
                                          (78.4263,87.7689,114.8958), swapRB=False)
        age_net.setInput(face_blob)
        pred = age_net.forward()[0]
        age_id   = int(np.argmax(pred))
        age_conf = float(np.max(pred))
        age_label = AGE_BUCKETS[age_id] if 0 <= age_id < len(AGE_BUCKETS) else str(age_id)

        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.putText(annotated, f"{age_label} {age_conf:.2f}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        results.append({"bbox_xyxy":[x1,y1,x2,y2], "age_bucket":age_label, "confidence":age_conf})

    save_outputs(annotated, Path(image_path), "age",
                 {"image": image_path, "detections": results})


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["ocr", "emotion", "objects", "age"])
    ap.add_argument("--image", required=True, help="Path to input image")

    # Objects params
    ap.add_argument("--conf", type=float, default=0.35, help="[objects] confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="[objects] IoU threshold")
    ap.add_argument("--model", default=None, help="[objects] path to ONNX model")
    ap.add_argument("--debug", action="store_true", help="[objects] extra prints")

    args = ap.parse_args()

    start = time.time()

    if args.task == "ocr":
        run_ocr(args.image)
    elif args.task == "emotion":
        run_emotion_deepface_mtcnn(args.image)
    elif args.task == "objects":
        run_objects(args.image, conf_th=args.conf, iou_th=args.iou,
                    model_override=args.model, debug=args.debug)
    elif args.task == "age":
        run_age(args.image)

    print(f"Total elapsed time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
