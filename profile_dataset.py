import os
import glob

# Search in the python-engine dataset splits as well as local fallback
LABEL_DIRS = [
    "python-engine/dataset/labels/train",
    "python-engine/dataset/labels/val",
    "python-engine/dataset/labels/test",
    "python-engine/dataset/labels",
    "dataset/labels/train",
    "dataset/labels/val",
    "dataset/labels/test",
    "dataset/labels",
]

IMG_W, IMG_H = 640, 640

box_areas = []
small_count = medium_count = large_count = 0

label_files = []
for d in LABEL_DIRS:
    if os.path.exists(d):
        label_files.extend(glob.glob(os.path.join(d, "*.txt")))

# Deduplicate
label_files = list(dict.fromkeys(os.path.abspath(f) for f in label_files))

for file_path in label_files:
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    w_px = float(parts[3]) * IMG_W
                    h_px = float(parts[4]) * IMG_H
                    area = w_px * h_px
                    box_areas.append(area)
                    
                    if area < 32**2:
                        small_count += 1
                    elif area <= 96**2:
                        medium_count += 1
                    else:
                        large_count += 1
                except ValueError:
                    continue

total_boxes = len(box_areas)
print(f"Total Label Files Analyzed: {len(label_files)}")
print(f"Total Bounding Boxes: {total_boxes}")
if total_boxes > 0:
    print(f"Small (<32x32 px):  {small_count} ({small_count/total_boxes*100:.1f}%)")
    print(f"Medium (32-96 px): {medium_count} ({medium_count/total_boxes*100:.1f}%)")
    print(f"Large (>96x96 px):  {large_count} ({large_count/total_boxes*100:.1f}%)")
