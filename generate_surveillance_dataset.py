"""
generate_surveillance_dataset.py — Creates a Surveillance-Scale Variant of the YOLO Dataset

Transforms close-up bottle/alcohol images into distant, surveillance-framed scenes by:
1. Downscaling images to physically shrink targets (e.g. scale factor ~0.10 - 0.15x).
2. Padding onto a 640x640 neutral gray canvas (114, 114, 114).
3. Placing the scaled scene in random quadrants, center, or random offsets.
4. Recalculating normalized YOLO bounding box coordinates (class, xc, yc, w, h).
5. Saving to dataset_surveillance/ (preserving train/val/test splits).
6. Calculating and printing the scale distribution to verify the majority are Small (<32x32 px).
"""

import os
import glob
import cv2
import yaml
import shutil
import random
import argparse
import numpy as np
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent

def find_source_dataset():
    candidates = [
        ROOT_DIR / "python-engine" / "dataset",
        ROOT_DIR / "dataset",
    ]
    for c in candidates:
        if (c / "images").exists() and (c / "labels").exists():
            return c
    raise FileNotFoundError("Could not locate source dataset directory with images/ and labels/.")

def generate_surveillance_dataset(
    src_dir: Path,
    dest_dir: Path,
    min_scale: float = 0.10,
    max_scale: float = 0.14,
    canvas_size: int = 640,
    bg_color: tuple = (114, 114, 114),
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("GENERATING SURVEILLANCE-SCALE DATASET VARIANT")
    print("=" * 70)
    print(f"Source Directory:      {src_dir}")
    print(f"Destination Directory: {dest_dir}")
    print(f"Scale Factor Range:    {min_scale:.2f}x - {max_scale:.2f}x")
    print(f"Canvas Resolution:     {canvas_size}x{canvas_size}")
    print(f"Canvas Neutral Gray:   {bg_color}\n")
    
    # Identify splits
    splits = []
    for split_candidate in ["train", "val", "test"]:
        if (src_dir / "images" / split_candidate).exists():
            splits.append(split_candidate)
            
    if not splits:
        # Flat structure fallback
        splits = [""]

    total_images_processed = 0
    total_boxes_processed = 0

    # Ensure clean output directories
    if dest_dir.exists():
        print(f"Cleaning existing directory: {dest_dir}")
        shutil.rmtree(dest_dir)

    for split in splits:
        dest_img_dir = dest_dir / "images" / split
        dest_lbl_dir = dest_dir / "labels" / split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        src_img_dir = src_dir / "images" / split
        src_lbl_dir = src_dir / "labels" / split
        
        img_paths = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.jpeg")) + list(src_img_dir.glob("*.png"))
        print(f"Processing split '{split or 'root'}': {len(img_paths)} images...")
        
        for img_path in img_paths:
            # Read original image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            orig_h, orig_w = img.shape[:2]
            
            # Select random scale factor within desired range
            scale = random.uniform(min_scale, max_scale)
            scaled_w = int(round(orig_w * scale))
            scaled_h = int(round(orig_h * scale))
            
            # Ensure at least 1 pixel
            scaled_w = max(1, min(canvas_size, scaled_w))
            scaled_h = max(1, min(canvas_size, scaled_h))
            
            # Resize image
            scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            
            # Choose placement on canvas:
            # Random placement simulation of distant camera framing
            max_x_offset = canvas_size - scaled_w
            max_y_offset = canvas_size - scaled_h
            
            # Strategy: Randomly select between quadrant corners, center, or uniform random
            mode = random.choice(["corner_tl", "corner_tr", "corner_bl", "corner_br", "center", "random"])
            if mode == "corner_tl":
                offset_x, offset_y = 0, 0
            elif mode == "corner_tr":
                offset_x, offset_y = max_x_offset, 0
            elif mode == "corner_bl":
                offset_x, offset_y = 0, max_y_offset
            elif mode == "corner_br":
                offset_x, offset_y = max_x_offset, max_y_offset
            elif mode == "center":
                offset_x, offset_y = max_x_offset // 2, max_y_offset // 2
            else:
                offset_x = random.randint(0, max_x_offset) if max_x_offset > 0 else 0
                offset_y = random.randint(0, max_y_offset) if max_y_offset > 0 else 0
                
            # Create neutral canvas
            canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
            canvas[offset_y:offset_y + scaled_h, offset_x:offset_x + scaled_w] = scaled_img
            
            # Save new image
            out_img_path = dest_img_dir / img_path.name
            cv2.imwrite(str(out_img_path), canvas)
            
            # Process corresponding label
            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            out_lbl_lines = []
            
            if lbl_path.exists():
                with open(lbl_path, "r") as f_in:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = parts[0]
                            xc = float(parts[1])
                            yc = float(parts[2])
                            bw = float(parts[3])
                            bh = float(parts[4])
                            
                            # Original absolute box coordinates in source image
                            box_w_px = bw * orig_w
                            box_h_px = bh * orig_h
                            box_xc_px = xc * orig_w
                            box_yc_px = yc * orig_h
                            
                            # Transformed box coordinates on scaled canvas
                            new_box_w_px = box_w_px * scale
                            new_box_h_px = box_h_px * scale
                            new_box_xc_px = offset_x + (box_xc_px * scale)
                            new_box_yc_px = offset_y + (box_yc_px * scale)
                            
                            # Convert to normalized coordinates [0.0, 1.0] relative to canvas
                            new_xc = new_box_xc_px / canvas_size
                            new_yc = new_box_yc_px / canvas_size
                            new_bw = new_box_w_px / canvas_size
                            new_bh = new_box_h_px / canvas_size
                            
                            # Clamp bounding box corners within [0.0, 1.0]
                            x1 = max(0.0, min(1.0, new_xc - (new_bw / 2.0)))
                            y1 = max(0.0, min(1.0, new_yc - (new_bh / 2.0)))
                            x2 = max(0.0, min(1.0, new_xc + (new_bw / 2.0)))
                            y2 = max(0.0, min(1.0, new_yc + (new_bh / 2.0)))
                            
                            clamped_xc = (x1 + x2) / 2.0
                            clamped_yc = (y1 + y2) / 2.0
                            clamped_bw = x2 - x1
                            clamped_bh = y2 - y1
                            
                            if clamped_bw > 0 and clamped_bh > 0:
                                out_lbl_lines.append(
                                    f"{cls_id} {clamped_xc:.6f} {clamped_yc:.6f} {clamped_bw:.6f} {clamped_bh:.6f}"
                                )
                                total_boxes_processed += 1
                                
            out_lbl_path = dest_lbl_dir / f"{img_path.stem}.txt"
            with open(out_lbl_path, "w") as f_out:
                if out_lbl_lines:
                    f_out.write("\n".join(out_lbl_lines) + "\n")
                    
            total_images_processed += 1

    # Create dataset_surveillance/data.yaml
    yaml_content = {
        "path": str(dest_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "alcohol",
            1: "no-alcohol"
        }
    }
    with open(dest_dir / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"\nSuccessfully generated {total_images_processed} surveillance-scale images with {total_boxes_processed} boxes.")
    print(f"data.yaml generated at: {dest_dir / 'data.yaml'}\n")

def verify_scale_distribution(dest_dir: Path, canvas_size: int = 640):
    print("=" * 70)
    print("VERIFICATION: BOUNDING BOX SCALE DISTRIBUTION (SURVEILLANCE DATASET)")
    print("=" * 70)
    
    label_files = list(dest_dir.glob("labels/**/*.txt"))
    if not label_files:
        label_files = list(dest_dir.glob("labels/*.txt"))
        
    box_areas = []
    small_count = 0   # < 32^2 = 1024 px^2
    medium_count = 0  # 32^2 to 96^2 = 1024 to 9216 px^2
    large_count = 0   # > 96^2 = 9216 px^2
    
    for lbl_file in label_files:
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    bw = float(parts[3]) * canvas_size
                    bh = float(parts[4]) * canvas_size
                    area = bw * bh
                    box_areas.append(area)
                    
                    if area < 32**2:
                        small_count += 1
                    elif area <= 96**2:
                        medium_count += 1
                    else:
                        large_count += 1
                        
    total = len(box_areas)
    print(f"Total Label Files:     {len(label_files)}")
    print(f"Total Bounding Boxes:  {total}")
    if total > 0:
        pct_small = (small_count / total) * 100
        pct_med = (medium_count / total) * 100
        pct_large = (large_count / total) * 100
        print(f"Small (<32x32 px):     {small_count:<5} ({pct_small:.1f}%)")
        print(f"Medium (32-96 px):    {medium_count:<5} ({pct_med:.1f}%)")
        print(f"Large (>96x96 px):     {large_count:<5} ({pct_large:.1f}%)")
        print("=" * 70)
        if pct_small > 50.0:
            print("SUCCESS: The majority of targets now fall into the 'Small' category (< 1024 px^2)!")
        else:
            print("INFO: Target distribution shifted towards small/medium.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Surveillance-Scale YOLO Dataset")
    parser.add_argument("--src", type=str, default=None, help="Path to source dataset directory")
    parser.add_argument("--dest", type=str, default="dataset_surveillance", help="Path to destination dataset directory")
    parser.add_argument("--min-scale", type=float, default=0.10, help="Minimum scale factor (default: 0.10)")
    parser.add_argument("--max-scale", type=float, default=0.14, help="Maximum scale factor (default: 0.14)")
    parser.add_argument("--canvas", type=int, default=640, help="Canvas size in pixels (default: 640)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    src_path = Path(args.src) if args.src else find_source_dataset()
    dest_path = Path(args.dest) if os.path.isabs(args.dest) else ROOT_DIR / args.dest
    
    generate_surveillance_dataset(
        src_dir=src_path,
        dest_dir=dest_path,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        canvas_size=args.canvas,
        seed=args.seed
    )
    
    verify_scale_distribution(dest_path, canvas_size=args.canvas)
