"""
train_alcohol_model.py — Fine-tune YOLOv8 on Alcohol Detection Dataset (nitro dataset split)

Usage:
    python train_alcohol_model.py --epochs 50 --imgsz 640
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

DATASET_DIR = Path(__file__).parent / "dataset"
OUTPUT_MODEL = Path(__file__).parent / "yolov8n_alcohol.pt"

def prepare_dataset_splits(data_source: Path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split images/labels into train, val, and test directories.
    """
    images_dir = data_source / "images"
    labels_dir = data_source / "labels"
    
    if not images_dir.exists():
        print(f"Directory {images_dir} does not exist. Creating default split layout...")
        os.makedirs(DATASET_DIR / "images" / "train", exist_ok=True)
        os.makedirs(DATASET_DIR / "images" / "val", exist_ok=True)
        os.makedirs(DATASET_DIR / "images" / "test", exist_ok=True)
        os.makedirs(DATASET_DIR / "labels" / "train", exist_ok=True)
        os.makedirs(DATASET_DIR / "labels" / "val", exist_ok=True)
        os.makedirs(DATASET_DIR / "labels" / "test", exist_ok=True)
        return

    all_images = [f for f in images_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    random.shuffle(all_images)
    
    n_total = len(all_images)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    
    train_imgs = all_images[:n_train]
    val_imgs   = all_images[n_train:n_train + n_val]
    test_imgs  = all_images[n_train + n_val:]
    
    splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
    
    for split_name, img_list in splits.items():
        split_img_dir = DATASET_DIR / "images" / split_name
        split_lbl_dir = DATASET_DIR / "labels" / split_name
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)
        
        for img_path in img_list:
            shutil.copy(img_path, split_img_dir / img_path.name)
            label_file = labels_dir / f"{img_path.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, split_lbl_dir / label_file.name)

def generate_yaml():
    yaml_content = {
        "path": str(DATASET_DIR.absolute()),
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

def train_model(epochs=50, imgsz=640):
    yaml_path = generate_yaml()
    print(f"Loading baseline model yolov8n.pt for fine-tuning...")
    model = YOLO("yolov8n.pt")
    
    print(f"Starting fine-tuning on alcohol dataset ({epochs} epochs)...")
    results = model.train(data=str(yaml_path), epochs=epochs, imgsz=imgsz, project="runs", name="alcohol_finetune", exist_ok=True)
    
    best_weights = Path("runs/alcohol_finetune/weights/best.pt")
    if best_weights.exists():
        shutil.copy(best_weights, OUTPUT_MODEL)
        print(f"Exported fine-tuned weights to {OUTPUT_MODEL}")
    else:
        print("Training complete.")

if __name__ == "__main__":
    prepare_dataset_splits(DATASET_DIR)
    train_model(epochs=50)
