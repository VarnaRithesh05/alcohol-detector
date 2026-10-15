"""
train_ablation_study.py — Automated 4-Model Ablation Study Suite

Performs sequential training and granular evaluation for the research paper:
1. Baseline: Standard YOLOv8n + CIoU Loss
2. Loss Ablation: Standard YOLOv8n + Continuous NWD Loss
3. Architecture Ablation: Custom YOLOv8-P2 (Stride-4 shallow head) + CIoU Loss
4. Proposed Full Model: Custom YOLOv8-P2 (Stride-4 shallow head) + Continuous NWD Loss

Evaluates each trained best.pt against the test split of dataset_surveillance
to extract Overall mAP@0.5 and Small-Target AP (APs for targets < 32^2 px).
"""

import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.utils.loss as uloss

# Ensure python-engine is on sys.path to import nwd_loss
ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python-engine")
if ENGINE_DIR not in sys.path:
    sys.path.append(ENGINE_DIR)

from nwd_loss import compute_nwd

# Global flag controlling whether NWD is active in BboxLoss
USE_NWD_LOSS = False
_ORIGINAL_BBOXLOSS_FORWARD = uloss.BboxLoss.forward

def patched_bboxloss_forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
    imgsz: torch.Tensor,
    stride: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Substitutes standard CIoU regression with Continuous Normalized Wasserstein Distance (NWD)
    when USE_NWD_LOSS is enabled. Retains standard CIoU otherwise.
    """
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

    if USE_NWD_LOSS:
        # Normalized Wasserstein Distance continuous distribution loss
        # Constant C=12.8 tuned for 640x640 surveillance-scale objects
        nwd = compute_nwd(pred_bboxes[fg_mask], target_bboxes[fg_mask], constant=12.8)
        loss_box = ((1.0 - nwd) * weight).sum() / target_scores_sum
    else:
        # Standard Complete IoU (CIoU)
        iou = uloss.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum

    # Distribution Focal Loss (DFL)
    if self.dfl_loss:
        target_ltrb = uloss.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        target_ltrb = uloss.bbox2dist(anchor_points, target_bboxes)
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss_dfl = (
            uloss.F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_box, loss_dfl

# Install patched forward on BboxLoss
uloss.BboxLoss.forward = patched_bboxloss_forward


def calc_iou(b1, b2):
    """Calculates 2D IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def compute_scale_ap(preds_list, gts_list, iou_thresh=0.5):
    """
    Computes Average Precision (AP@0.5) for a filtered subset of predictions and ground truths.
    Used for granular APs (<32^2 px) and APm (32^2 - 96^2 px).
    """
    if len(gts_list) == 0:
        return 0.0
    sorted_p = sorted(preds_list, key=lambda x: x["conf"], reverse=True)
    tp, fp = [], []
    matched = set()

    for p in sorted_p:
        img = p["img"]
        img_gts = [g for g in gts_list if g["img"] == img and g["cls"] == p["cls"]]
        best_iou, best_idx = -1.0, -1
        for i, g in enumerate(img_gts):
            if (img, i) in matched:
                continue
            iou = calc_iou(p["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_thresh:
            tp.append(1)
            fp.append(0)
            matched.add((img, best_idx))
        else:
            tp.append(0)
            fp.append(1)

    if not tp:
        return 0.0

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


def evaluate_granular(weights_path: str, data_yaml: str, imgsz: int = 640, conf: float = 0.20, iou: float = 0.50):
    """
    Performs standard Ultralytics validation + granular scale-stratified APs calculation (<32^2 px)
    on the test split of the dataset.
    """
    if not os.path.exists(weights_path):
        print(f"Warning: Weights not found at {weights_path}")
        return 0.0, 0.0, 0.0

    model = YOLO(weights_path)

    # 1. Standard Ultralytics test validation
    val_res = model.val(
        data=data_yaml,
        split="test",
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False
    )
    overall_map50 = float(val_res.box.map50)

    # 2. Extract test dataset directory
    base_dir = os.path.dirname(os.path.abspath(data_yaml))
    test_img_dir = os.path.join(base_dir, "images", "test")
    test_lbl_dir = os.path.join(base_dir, "labels", "test")

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
            with open(lbl_p, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) >= 5:
                        cls_id = int(p[0])
                        xc, yc, bw, bh = map(float, p[1:5])
                        x1 = (xc - bw / 2.0) * w
                        y1 = (yc - bh / 2.0) * h
                        x2 = (xc + bw / 2.0) * w
                        y2 = (yc + bh / 2.0) * h
                        area = (bw * w) * (bh * h)
                        scale = "small" if area < 32**2 else ("medium" if area <= 96**2 else "large")
                        gts_by_scale[scale].append({"bbox": [x1, y1, x2, y2], "cls": cls_id, "img": stem})

        res = model(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
        for b in res.boxes:
            b_conf = float(b.conf[0])
            b_cls = int(b.cls[0])
            box = b.xyxy[0].tolist()
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            area = bw * bh
            scale = "small" if area < 32**2 else ("medium" if area <= 96**2 else "large")
            preds.append({"bbox": box, "conf": b_conf, "cls": b_cls, "img": stem, "scale": scale})

    ap_s = compute_scale_ap([p for p in preds if p["scale"] == "small"], gts_by_scale["small"])
    ap_m = compute_scale_ap([p for p in preds if p["scale"] == "medium"], gts_by_scale["medium"])

    return overall_map50, ap_s, ap_m


def run_ablation_study(
    data_yaml: str = "dataset_surveillance/data.yaml",
    epochs: int = 40,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "auto",
    eval_only: bool = False,
    quick_test: bool = False,
    selected_models: str = None
):
    global USE_NWD_LOSS

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device in ["0", "cuda", "cuda:0", 0] and not torch.cuda.is_available():
        print(f"\n[!] WARNING: CUDA device '{device}' was requested, but torch.cuda.is_available() is False.")
        print("[!] PyTorch is currently CPU-only. Falling back automatically to device='cpu'.")
        print("[!] To use your physical NVIDIA GPU, install CUDA PyTorch via:")
        print("    pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121\n")
        device = "cpu"

    if quick_test:
        epochs = 1
        print(">>> Quick test mode enabled: epochs set to 1 <<<")

    print("=" * 80)
    print("RESEARCH PAPER ABLATION STUDY SUITE")
    print(f"Dataset:      {data_yaml}")
    print(f"Epochs:       {epochs}")
    print(f"Batch Size:   {batch}")
    print(f"Image Size:   {imgsz}")
    print(f"Device:       {device} (CUDA Available: {torch.cuda.is_available()})")
    print("=" * 80)

    # 4 Experimental Configurations
    # 1. Baseline: Standard YOLOv8n + Standard CIoU
    # 2. Loss Ablation: Standard YOLOv8n + Custom NWD
    # 3. Architecture Ablation: Custom yolov8_p2_nwd.yaml (Stride-4 head) + Standard CIoU
    # 4. Proposed Full Model: Custom yolov8_p2_nwd.yaml + Custom NWD
    all_configs = [
        {
            "name": "Baseline (YOLOv8n + CIoU)",
            "key": "baseline_ciou",
            "model_cfg": "yolov8n.pt",
            "pretrained": True,
            "use_nwd": False,
        },
        {
            "name": "Loss Ablation (YOLOv8n + NWD)",
            "key": "loss_ablation_nwd",
            "model_cfg": "yolov8n.pt",
            "pretrained": True,
            "use_nwd": True,
        },
        {
            "name": "Architecture Ablation (YOLOv8-P2 + CIoU)",
            "key": "arch_ablation_p2",
            "model_cfg": "python-engine/models/yolov8_p2_nwd.yaml",
            "pretrained_weights": "yolov8n.pt",
            "pretrained": False,
            "use_nwd": False,
        },
        {
            "name": "Proposed Full Model (YOLOv8-P2 + NWD)",
            "key": "proposed_full_p2_nwd",
            "model_cfg": "python-engine/models/yolov8_p2_nwd.yaml",
            "pretrained_weights": "yolov8n.pt",
            "pretrained": False,
            "use_nwd": True,
        },
    ]

    if selected_models:
        keys = [k.strip().lower() for k in selected_models.split(",")]
        configs = [c for c in all_configs if c["key"] in keys or any(k in c["key"] for k in keys)]
    else:
        configs = all_configs

    results = []
    output_dir = "runs/ablation"
    os.makedirs(output_dir, exist_ok=True)

    for idx, cfg in enumerate(configs, 1):
        print("\n" + "#" * 80)
        print(f"[{idx}/4] CONFIGURATION: {cfg['name']}")
        print(f"Architecture: {cfg['model_cfg']} | Loss: {'NWD Loss' if cfg['use_nwd'] else 'Standard CIoU'}")
        print("#" * 80)

        # Set NWD loss flag
        USE_NWD_LOSS = cfg["use_nwd"]

        save_name = cfg["key"]
        save_dir = os.path.join(os.path.abspath(output_dir), save_name)
        best_pt = os.path.join(save_dir, "weights", "best.pt")

        if not eval_only:
            # Initialize model
            if cfg.get("pretrained_weights"):
                # Custom architecture initialized with pretrained backbone weights
                model = YOLO(cfg["model_cfg"]).load(cfg["pretrained_weights"])
            else:
                model = YOLO(cfg["model_cfg"])

            start_t = time.time()
            model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                project=os.path.abspath(output_dir),
                name=save_name,
                exist_ok=True,
                verbose=True,
                patience=50,
                save=True,
                val=True
            )
            elapsed = time.time() - start_t
            print(f"Training for {cfg['name']} completed in {elapsed/60:.2f} minutes.")

            if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir"):
                actual_save_dir = str(model.trainer.save_dir)
                candidate_best = os.path.join(actual_save_dir, "weights", "best.pt")
                if os.path.exists(candidate_best):
                    best_pt = candidate_best
        else:
            print(f"Skipping training (eval_only mode). Looking for {best_pt}...")

        # If best.pt not generated, fallback to last.pt
        if not os.path.exists(best_pt):
            last_pt = os.path.join(save_dir, "weights", "last.pt")
            if os.path.exists(last_pt):
                best_pt = last_pt

        # Granular Evaluation on Test Split
        print(f"Running granular evaluation on {best_pt}...")
        map50, ap_s, ap_m = evaluate_granular(best_pt, data_yaml, imgsz=imgsz)

        print(f"Results for {cfg['name']}:")
        print(f"  Overall mAP@0.5:     {map50:.4f}")
        print(f"  Small Targets (APs): {ap_s:.4f}")
        print(f"  Medium Targets (APm): {ap_m:.4f}")

        results.append({
            "model_name": cfg["name"],
            "map50": map50,
            "ap_s": ap_s,
            "ap_m": ap_m,
            "weights": best_pt
        })

    # Print Summary Table
    print("\n" + "=" * 80)
    print("FINAL ABLATION STUDY RESULTS SUMMARY (TEST SPLIT)")
    print("=" * 80)
    header = f"{'Model Name':<42} | {'mAP@0.5':<10} | {'AP_s (<32^2 px)':<16} | {'AP_m (32-96^2 px)':<18}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model_name']:<42} | {r['map50']:<10.4f} | {r['ap_s']:<16.4f} | {r['ap_m']:<18.4f}")
    print("=" * 80)

    # Save to Markdown
    md_path = "ablation_study_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Study Results on Surveillance Dataset\n\n")
        f.write(f"- **Dataset**: `{data_yaml}` (Surveillance-scale micro target variant)\n")
        f.write(f"- **Epochs**: {epochs} | **Batch**: {batch} | **Image Size**: {imgsz}x{imgsz}\n\n")
        f.write("| Model Name | Architecture | Loss Function | Overall mAP@0.5 | Small-Target AP ($AP_s$) | Medium-Target AP ($AP_m$) |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: |\n")
        for r in results:
            arch = "YOLOv8-P2 (Stride-4)" if "P2" in r["model_name"] else "Standard YOLOv8n"
            loss_type = "Continuous NWD" if "NWD" in r["model_name"] else "Standard CIoU"
            f.write(f"| **{r['model_name']}** | {arch} | {loss_type} | **{r['map50']:.4f}** | **{r['ap_s']:.4f}** | {r['ap_m']:.4f} |\n")
    print(f"\nMarkdown summary saved to: {md_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4-Model Ablation Study for YOLOv8 Small Object Detection")
    parser.add_argument("--data", type=str, default="dataset_surveillance/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda:0, auto)")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on existing trained models")
    parser.add_argument("--quick-test", action="store_true", help="Run 1 epoch test run to verify pipeline integrity")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys to run: baseline_ciou, loss_ablation_nwd, arch_ablation_p2, proposed_full_p2_nwd")

    args = parser.parse_args()
    run_ablation_study(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        eval_only=args.eval_only,
        quick_test=args.quick_test,
        selected_models=args.models
    )
