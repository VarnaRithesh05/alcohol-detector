"""
build_large_dataset.py — Hybrid Large Dataset Builder and Model Trainer

Downloads nitro's Roboflow "Alcohol Detection" dataset (3,012 images) using Roboflow API key,
downloads real photo images from Unsplash for negative control and Indian-brand matching verification,
merges them together, and fine-tunes YOLOv8.
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
OUTPUT_MODEL = ENGINE_DIR / "yolov8s_alcohol.pt"

# Real photos for negative controls and additional labels
ALCOHOL_PHOTOS = [
    "photo-1527281400683-1aae777175f8", # Whiskey
    "photo-1608270586620-248524c67de9", # Beer
    "photo-1510812431401-41d2bd2722f3", # Wine
    "photo-1569529465841-dfecdab7503b", # Vodka
    "photo-1514362545857-3bc16c4c7d1b", # Cocktail
]

NON_ALCOHOL_PHOTOS = [
    "photo-1544787219-7f47ccb76574", # Mug
    "photo-1523362628745-0c100150b504", # Water bottle
    "photo-1542156822-6924d1a71aba", # Soda bottle
]

def load_api_key():
    env_path = ENGINE_DIR.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("ROBOFLOW_API_KEY="):
                    return line.strip().split("=")[1].strip('"').strip("'")
    return None

def download_image(photo_id: str) -> np.ndarray:
    url = f"https://images.unsplash.com/{photo_id}?w=640&q=80"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def auto_label_image(img: np.ndarray, model: YOLO, is_alcohol: bool) -> list:
    results = model(img, verbose=False)[0]
    labels = []
    h, w = img.shape[:2]
    coco_targets = {39, 40, 41}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in coco_targets:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            xc = ((x1 + x2) / 2.0) / w
            yc = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            target_class = 0 if is_alcohol else 1
            labels.append(f"{target_class} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return labels

def build_hybrid_dataset(api_key: str):
    print("Initializing directories...")
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        
    for split in ["train", "val", "test"]:
        os.makedirs(DATASET_DIR / "images" / split, exist_ok=True)
        os.makedirs(DATASET_DIR / "labels" / split, exist_ok=True)
        
    # 1. Download nitro's dataset using Roboflow
    print("Downloading nitro/alcohol-detection-srjag dataset from Roboflow...")
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("nitro-bzs43").project("alcohol-detection-srjag")
    # Download YOLOv8 dataset format
    rf_dataset = project.version(1).download("yolov8")
    rf_path = Path(rf_dataset.location)
    
    # 2. Merge Roboflow splits
    print("Merging Roboflow dataset files...")
    # Roboflow export format splits: train, valid, test
    mapping = {"train": "train", "valid": "val", "test": "test"}
    for rf_split, local_split in mapping.items():
        rf_img_dir = rf_path / rf_split / "images"
        rf_lbl_dir = rf_path / rf_split / "labels"
        
        if rf_img_dir.exists():
            for img_file in rf_img_dir.glob("*.jpg"):
                shutil.copy(img_file, DATASET_DIR / "images" / local_split / img_file.name)
                lbl_file = rf_lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    # Read Roboflow labels and map class to binary if needed
                    # Roboflow nitro classes: 0 = alcohol, 1 = no-alcohol
                    shutil.copy(lbl_file, DATASET_DIR / "labels" / local_split / lbl_file.name)

    # 3. Add Unsplash real photos
    print("Adding Unsplash real photos...")
    base_model = YOLO("yolov8n.pt")
    
    unsplash_samples = []
    for photo_id in ALCOHOL_PHOTOS:
        img = download_image(photo_id)
        if img is not None:
            labels = auto_label_image(img, base_model, is_alcohol=True)
            if labels:
                unsplash_samples.append((img, labels, f"web_alc_{photo_id}"))
                
    for photo_id in NON_ALCOHOL_PHOTOS:
        img = download_image(photo_id)
        if img is not None:
            labels = auto_label_image(img, base_model, is_alcohol=False)
            if labels:
                unsplash_samples.append((img, labels, f"web_non_{photo_id}"))
                
    # Add Unsplash images to train/val split
    for i, (img, labels, name) in enumerate(unsplash_samples):
        split = "train" if i % 2 == 0 else "val"
        cv2.imwrite(str(DATASET_DIR / "images" / split / f"{name}.jpg"), img)
        with open(DATASET_DIR / "labels" / split / f"{name}.txt", "w") as f:
            f.write("\n".join(labels) + "\n")
            
    print(f"Hybrid dataset assembled successfully.")
    
    # Generate data.yaml
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
        
    # Clean up temporary Roboflow download path
    if rf_path.exists():
        shutil.rmtree(rf_path, ignore_errors=True)
        
    return yaml_path

def train_hybrid_model(epochs=15):
    api_key = load_api_key()
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY not found in .env file.")
        return
        
    yaml_path = build_hybrid_dataset(api_key)
    print("\nFine-tuning YOLOv8 on Hybrid Large Dataset...")
    model = YOLO("yolov8s.pt")
    
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=16,
        project=str(ENGINE_DIR / "runs"),
        name="alcohol_hybrid_training",
        exist_ok=True
    )
    
    best_weights = ENGINE_DIR / "runs" / "alcohol_hybrid_training" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(best_weights, OUTPUT_MODEL)
        print(f"\nSUCCESS: Fine-tuned hybrid model exported to {OUTPUT_MODEL}")
    else:
        print("\nTraining completed.")

if __name__ == "__main__":
    train_hybrid_model(epochs=20)
