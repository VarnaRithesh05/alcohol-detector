"""
build_dataset_and_train.py — Dataset Creator and YOLOv8 Alcohol Model Trainer

Downloads/generates an annotated alcohol dataset with train/val/test splits (80/10/10)
and fine-tunes YOLOv8n to produce yolov8n_alcohol.pt.
"""

import os
import cv2
import yaml
import shutil
import random
import urllib.request
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ENGINE_DIR = Path(__file__).parent
DATASET_DIR = ENGINE_DIR / "dataset"
OUTPUT_MODEL = ENGINE_DIR / "yolov8n_alcohol.pt"

# Public domain / CC sample images for training
SAMPLE_SOURCES = [
    # (url, label_class, bbox_rel: (x_center, y_center, w, h), name)
    ("https://images.unsplash.com/photo-1527281400683-1aae777175f8?w=640", 0, (0.5, 0.5, 0.4, 0.8), "whiskey_bottle_1"),
    ("https://images.unsplash.com/photo-1608270586620-248524c67de9?w=640", 0, (0.5, 0.55, 0.35, 0.75), "beer_bottle_1"),
    ("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=640", 0, (0.5, 0.5, 0.3, 0.85), "wine_bottle_1"),
    ("https://images.unsplash.com/photo-1569529465841-dfecdab7503b?w=640", 0, (0.48, 0.52, 0.38, 0.8), "vodka_bottle_1"),
    ("https://images.unsplash.com/photo-1514362545857-3bc16c4c7d1b?w=640", 0, (0.5, 0.5, 0.45, 0.7), "cocktail_glass_1"),
    ("https://images.unsplash.com/photo-1544787219-7f47ccb76574?w=640", 1, (0.5, 0.5, 0.5, 0.6), "coffee_mug_1"),
    ("https://images.unsplash.com/photo-1523362628745-0c100150b504?w=640", 1, (0.5, 0.5, 0.3, 0.75), "water_bottle_1"),
]

def generate_synthetic_samples(count=40):
    """Generates varied synthetic bottle/container samples with bounding box labels."""
    samples = []
    for i in range(count):
        img = np.full((480, 640, 3), random.randint(180, 240), dtype=np.uint8)
        is_alcohol = random.choice([True, True, False]) # 66% alcohol, 33% non-alcohol
        
        # Add background noise/shapes
        for _ in range(5):
            cv2.circle(img, (random.randint(0, 640), random.randint(0, 480)), random.randint(10, 50), (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)), -1)
            
        bx = random.randint(180, 420)
        by = random.randint(100, 300)
        bw = random.randint(80, 160)
        bh = random.randint(150, 280)
        
        if is_alcohol:
            # Draw bottle shape
            color = random.choice([(40, 80, 160), (30, 120, 50), (20, 40, 90)])
            cv2.rectangle(img, (bx - bw//2, by - bh//4), (bx + bw//2, by + bh//2), color, -1)
            cv2.rectangle(img, (bx - bw//6, by - bh//2), (bx + bw//6, by - bh//4), color, -1)
            # Label
            cv2.rectangle(img, (bx - bw//3, by - bh//8), (bx + bw//3, by + bh//4), (240, 240, 240), -1)
            brand = random.choice(["KINGFISHER", "OLD MONK", "BACARDI", "BIRA 91", "ROYAL STAG", "ABSOLUT"])
            cv2.putText(img, brand, (bx - bw//3 + 5, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 120), 1)
            cls_id = 0
        else:
            # Non-alcohol (mug / glass)
            color = random.choice([(180, 140, 100), (200, 200, 200), (100, 150, 180)])
            cv2.rectangle(img, (bx - bw//2, by - bh//3), (bx + bw//2, by + bh//3), color, -1)
            cv2.ellipse(img, (bx + bw//2, by), (bw//4, bh//4), 0, 0, 360, color, 8)
            cls_id = 1
            
        # Convert to YOLO format (class, x_center, y_center, width, height)
        xc = bx / 640.0
        yc = by / 480.0
        w_norm = bw / 640.0
        h_norm = bh / 480.0
        
        samples.append((img, cls_id, (xc, yc, w_norm, h_norm), f"syn_{i:03d}"))
    return samples

def build_dataset():
    print("Building alcohol dataset...")
    # Clear old dataset
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        
    for split in ["train", "val", "test"]:
        os.makedirs(DATASET_DIR / "images" / split, exist_ok=True)
        os.makedirs(DATASET_DIR / "labels" / split, exist_ok=True)
        
    all_data = []
    
    # 1. Download real web samples
    raw_dir = DATASET_DIR / "raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    for url, cls_id, bbox, name in SAMPLE_SOURCES:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    # Resize for uniform dataset training
                    img_resized = cv2.resize(img, (640, 480))
                    all_data.append((img_resized, cls_id, bbox, name))
        except Exception as e:
            print(f"Skipping {name} (network download timeout): {e}")

    # 2. Add synthetic sample frames
    syn_samples = generate_synthetic_samples(count=50)
    all_data.extend(syn_samples)
    
    # Shuffle and split 80/10/10
    random.shuffle(all_data)
    n_total = len(all_data)
    n_train = int(n_total * 0.8)
    n_val   = int(n_total * 0.1)
    
    train_set = all_data[:n_train]
    val_set   = all_data[n_train:n_train + n_val]
    test_set  = all_data[n_train + n_val:]
    
    splits = {"train": train_set, "val": val_set, "test": test_set}
    
    for split_name, items in splits.items():
        for img, cls_id, bbox, name in items:
            img_path = DATASET_DIR / "images" / split_name / f"{name}.jpg"
            lbl_path = DATASET_DIR / "labels" / split_name / f"{name}.txt"
            
            cv2.imwrite(str(img_path), img)
            xc, yc, w_norm, h_norm = bbox
            with open(lbl_path, "w") as f:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"Dataset generated: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test images.")
    
    # Create data.yaml
    yaml_content = {
        "path": str(DATASET_DIR.absolute()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "alcohol",
            1: "no-alcohol"
        }
    }
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    return yaml_path

def train_alcohol_model(epochs=15):
    yaml_path = build_dataset()
    print(f"\nStarting YOLOv8 fine-tuning on alcohol dataset ({epochs} epochs)...")
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=8,
        project=str(ENGINE_DIR / "runs"),
        name="alcohol_training",
        exist_ok=True,
        verbose=True
    )
    
    best_weights = ENGINE_DIR / "runs" / "alcohol_training" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(best_weights, OUTPUT_MODEL)
        print(f"\nSUCCESS: Fine-tuned model exported to {OUTPUT_MODEL}")
    else:
        print("\nTraining completed. Weights saved in runs/alcohol_training/weights/")

if __name__ == "__main__":
    train_alcohol_model(epochs=15)
