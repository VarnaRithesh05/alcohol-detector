# Alcohol Detector Evaluation Results

## Dataset Split Information
- **Train Set**: 685 images
- **Validation Set**: 85 images
- **Test Set**: 87 images
- **Epochs Trained**: 15 epochs

## Quantitative Metrics Table

| Model | Conf Thresh | Precision | Recall | F1-Score | mAP@50 | Latency (ms) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock COCO YOLOv8n (Baseline) | 0.20 | 0.353 | 0.620 | 0.450 | 0.364 | 120.45 | 8.3 |
| Stock COCO YOLOv8n (Baseline) | 0.50 | 0.537 | 0.482 | 0.508 | 0.364 | 132.31 | 7.6 |
| Fine-Tuned YOLOv8n (Alcohol) | 0.20 | 0.767 | 0.723 | 0.744 | 0.697 | 102.32 | 9.8 |
| Fine-Tuned YOLOv8n (Alcohol) | 0.50 | 0.881 | 0.650 | 0.748 | 0.697 | 101.96 | 9.8 |