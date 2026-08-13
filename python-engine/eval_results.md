# Alcohol Detector Evaluation Results

## Dataset Split Information
- **Train Set**: 685 images
- **Validation Set**: 85 images
- **Test Set**: 87 images
- **Epochs Trained**: 40 epochs

## Quantitative Metrics Table

| Model | Conf Thresh | Precision | Recall | F1-Score | mAP@50 | Latency (ms) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock COCO YOLOv8n (Baseline) | 0.20 | 0.353 | 0.620 | 0.450 | 0.364 | 45.28 | 22.1 |
| Stock COCO YOLOv8n (Baseline) | 0.50 | 0.537 | 0.482 | 0.508 | 0.364 | 46.34 | 21.6 |
| Fine-Tuned YOLOv8n (Alcohol) | 0.20 | 0.770 | 0.854 | 0.810 | 0.829 | 42.27 | 23.7 |
| Fine-Tuned YOLOv8n (Alcohol) | 0.50 | 0.878 | 0.788 | 0.831 | 0.829 | 42.43 | 23.6 |

## Confusion Breakdowns

### Fine-Tuned YOLOv8n (Alcohol) @ conf=0.20
|              | Predicted Alcohol | Predicted No Detection |
|--------------|-------------------|--------------------------|
| Actual Alcohol   | TP = 117       | FN = 20                |
| Actual No-Alcohol| FP = 35       | (TN not meaningful here) |

### Fine-Tuned YOLOv8n (Alcohol) @ conf=0.50
|              | Predicted Alcohol | Predicted No Detection |
|--------------|-------------------|--------------------------|
| Actual Alcohol   | TP = 108       | FN = 29                |
| Actual No-Alcohol| FP = 15       | (TN not meaningful here) |


*Note: True Negative (TN) is not well-defined for object detection, as there is no fixed count of 'negative boxes' like there is in image classification. Hence, this is a detection confusion breakdown rather than a classification one.*