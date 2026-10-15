import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Resolve weights path (prioritize the 40-epoch fine-tuned model)
weights_candidates = [
    "python-engine/yolov8n_alcohol.pt",
    "runs/detect/train/weights/best.pt",
    "python-engine/runs/alcohol_real_training/weights/best.pt"
]
weights_path = None
for p in weights_candidates:
    if os.path.exists(p):
        weights_path = p
        break

if not weights_path:
    raise FileNotFoundError("Could not find trained YOLO weights.")

print(f"Loading model weights from: {weights_path}")
model = YOLO(weights_path)

data_yaml = "python-engine/dataset/data.yaml" if os.path.exists("python-engine/dataset/data.yaml") else "dataset/data.yaml"

# 2. Run standard Ultralytics validation
metrics = model.val(
    data=data_yaml,
    split="test",
    imgsz=640,
    conf=0.20,
    iou=0.50,
    verbose=False
)

# 3. Compute scale-stratified AP (Small, Medium, Large)
IMG_W, IMG_H = 640, 640

def calc_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

test_img_dir = "python-engine/dataset/images/test" if os.path.exists("python-engine/dataset/images/test") else "dataset/images/test"
test_lbl_dir = "python-engine/dataset/labels/test" if os.path.exists("python-engine/dataset/labels/test") else "dataset/labels/test"

test_imgs = glob.glob(os.path.join(test_img_dir, "*.*"))
preds = []
gts_by_scale = {"small": [], "medium": [], "large": []}

for img_p in test_imgs:
    stem = os.path.splitext(os.path.basename(img_p))[0]
    lbl_p = os.path.join(test_lbl_dir, f"{stem}.txt")
    frame = cv2.imread(img_p)
    if frame is None:
        continue
    h, w = frame.shape[:2]
    
    if os.path.exists(lbl_p):
        with open(lbl_p) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    cls_id = int(p[0])
                    xc, yc, bw, bh = map(float, p[1:5])
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    area = (bw * w) * (bh * h)
                    scale = "small" if area < 32**2 else ("medium" if area <= 96**2 else "large")
                    gts_by_scale[scale].append({"bbox": [x1, y1, x2, y2], "cls": cls_id, "img": stem})
                    
    res = model(frame, verbose=False)[0]
    for b in res.boxes:
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        box = b.xyxy[0].tolist()
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        area = bw * bh
        scale = "small" if area < 32**2 else ("medium" if area <= 96**2 else "large")
        preds.append({"bbox": box, "conf": conf, "cls": cls_id, "img": stem, "scale": scale})

def compute_ap(preds_list, gts_list):
    if len(gts_list) == 0:
        return 0.0
    sorted_p = sorted(preds_list, key=lambda x: x["conf"], reverse=True)
    tp, fp = [], []
    matched = set()
    
    for p in sorted_p:
        img = p["img"]
        img_gts = [g for g in gts_list if g["img"] == img and g["cls"] == p["cls"]]
        best_iou, best_idx = -1, -1
        for i, g in enumerate(img_gts):
            if (img, i) in matched:
                continue
            iou = calc_iou(p["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= 0.5:
            tp.append(1)
            fp.append(0)
            matched.add((img, best_idx))
        else:
            tp.append(0)
            fp.append(1)
            
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    prec = tp_c / (tp_c + fp_c + 1e-6)
    rec = tp_c / (len(gts_list) + 1e-6)
    
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

ap_s = compute_ap([p for p in preds if p["scale"] == "small"], gts_by_scale["small"])
ap_m = compute_ap([p for p in preds if p["scale"] == "medium"], gts_by_scale["medium"])
ap_l = compute_ap([p for p in preds if p["scale"] == "large"], gts_by_scale["large"])

print("\n--- Granular Evaluation Results ---")
print(f"Overall mAP@0.5:      {metrics.box.map50:.4f}")
print(f"Small Targets (APs):  {ap_s:.4f}")
print(f"Medium Targets (APm): {ap_m:.4f}")
print(f"Large Targets (APl):  {ap_l:.4f}")
