"""
build_real_dataset.py — Real Image Dataset Downloader and YOLOv8 Model Trainer

Downloads real photos of alcohol bottles, glasses, and negative containers from Unsplash.
Uses pre-trained YOLOv8 to auto-label them, builds split directories (80/10/10),
and fine-tunes YOLOv8 to generate yolov8n_alcohol.pt.
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

# High-quality real image IDs from Unsplash
ALCOHOL_PHOTOS = [
    "photo-1527281400683-1aae777175f8", # Whiskey bottle
    "photo-1608270586620-248524c67de9", # Beer bottle
    "photo-1510812431401-41d2bd2722f3", # Wine bottles
    "photo-1569529465841-dfecdab7503b", # Vodka bottle
    "photo-1470252649378-9c29740c9fa8", # Bar counter with bottles
    "photo-1514362545857-3bc16c4c7d1b", # Cocktail glass
    "photo-1563227812-0ea4c22e6cc8", # Beer cans
    "photo-1571115177098-24ec42ed204d", # Pouring wine
    "photo-1578985545062-69928b1d9587", # Liquor bottles shelf
    "photo-1568649929103-28ffbefcad1e", # Champagne glass
    "photo-1597290282695-edc43d0e7129", # Beer glasses
    "photo-1582819509237-d5b75f20ff7c", # Wine bottles shelf
    "photo-1600728614277-3e1b12b55f17", # Bar shelves
    "photo-1551538827-9c02e6c0f6f4", # Glass of beer
    "photo-1607623814075-e51df1bdc82f", # Whiskey pour
    "photo-1513558161293-cdaf765ed2fd", # Drinks on table
]

NON_ALCOHOL_PHOTOS = [
    "photo-1544787219-7f47ccb76574", # Coffee mug
    "photo-1523362628745-0c100150b504", # Water bottle
    "photo-1542156822-6924d1a71aba", # Soda bottle
    "photo-1576092768241-dec231879fc3", # Water glass
    "photo-1517256064527-09c53b2d0ec6", # Teacup
    "photo-1536935338788-846bb9981813", # Glass of juice
    "photo-1562184552-997c461abbe6", # Plastic cups
    "photo-1595981267035-7b04ca84a82d", # Soft drink cans
]

def download_image(photo_id: str) -> np.ndarray:
    """Downloads a photo from Unsplash by its ID."""
    url = f"https://images.unsplash.com/{photo_id}?w=640&q=80"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"Failed to download image {photo_id}: {e}")
        return None

def auto_label_image(img: np.ndarray, model: YOLO, is_alcohol: bool) -> list:
    """Uses pretrained YOLO to find bottles/glasses/cups and generate YOLO coordinates."""
    results = model(img, verbose=False)[0]
    labels = []
    h, w = img.shape[:2]
    
    # COCO classes: bottle=39, wine glass=40, cup=41
    coco_targets = {39, 40, 41}
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in coco_targets:
            xyxy = box.xyxy[0].tolist()
            # Normalize to YOLO format
            x1, y1, x2, y2 = xyxy
            xc = ((x1 + x2) / 2.0) / w
            yc = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            
            # Map detected container to target class:
            # 0 = alcohol (for bottles/wine glasses in alcohol photos)
            # 1 = no-alcohol (for non-alcohol photos)
            target_class = 0 if is_alcohol else 1
            labels.append(f"{target_class} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            
    return labels

def create_dataset():
    print("Initializing directories...")
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        
    for split in ["train", "val", "test"]:
        os.makedirs(DATASET_DIR / "images" / split, exist_ok=True)
        os.makedirs(DATASET_DIR / "labels" / split, exist_ok=True)
        
    # Load base model for auto-labeling
    base_model = YOLO("yolov8n.pt")
    
    all_samples = []
    
    print("\n--- Downloading and Auto-Labeling Alcohol Photos ---")
    for photo_id in ALCOHOL_PHOTOS:
        img = download_image(photo_id)
        if img is not None:
            labels = auto_label_image(img, base_model, is_alcohol=True)
            if labels:
                all_samples.append((img, labels, f"alc_{photo_id}"))
                print(f"Downloaded and labeled alc_{photo_id} ({len(labels)} labels)")
                
    print("\n--- Downloading and Auto-Labeling Non-Alcohol Photos ---")
    for photo_id in NON_ALCOHOL_PHOTOS:
        img = download_image(photo_id)
        if img is not None:
            labels = auto_label_image(img, base_model, is_alcohol=False)
            if labels:
                all_samples.append((img, labels, f"non_{photo_id}"))
                print(f"Downloaded and labeled non_{photo_id} ({len(labels)} labels)")

    # Shuffle and split safely ensuring at least 1 sample in validation and test
    random.shuffle(all_samples)
    n_total = len(all_samples)
    if n_total < 3:
        train_set = all_samples
        val_set = all_samples
        test_set = all_samples
    else:
        n_val   = max(1, int(n_total * 0.15))
        n_test  = max(1, int(n_total * 0.15))
        n_train = max(1, n_total - n_val - n_test)
        
        train_set = all_samples[:n_train]
        val_set   = all_samples[n_train:n_train + n_val]
        test_set  = all_samples[n_train + n_val:]
    
    splits = {"train": train_set, "val": val_set, "test": test_set}
    
    for split_name, items in splits.items():
        for img, labels, name in items:
            img_path = DATASET_DIR / "images" / split_name / f"{name}.jpg"
            lbl_path = DATASET_DIR / "labels" / split_name / f"{name}.txt"
            
            cv2.imwrite(str(img_path), img)
            with open(lbl_path, "w") as f:
                f.write("\n".join(labels) + "\n")
                
    print(f"\nReal dataset built: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test images.")
    
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

def train_model(epochs=15):
    yaml_path = create_dataset()
    print("\nStarting YOLOv8 fine-tuning on real downloaded image dataset...")
    model = YOLO("yolov8n.pt")
    
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=8,
        project=str(ENGINE_DIR / "runs"),
        name="alcohol_real_training",
        exist_ok=True
    )
    
    best_weights = ENGINE_DIR / "runs" / "alcohol_real_training" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(best_weights, OUTPUT_MODEL)
        print(f"\nSUCCESS: Fine-tuned model on real images exported to {OUTPUT_MODEL}")
    else:
        print("\nTraining completed.")

if __name__ == "__main__":
    train_model(epochs=15)
