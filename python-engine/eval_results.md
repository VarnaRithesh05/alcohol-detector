# Alcohol Detector Evaluation Results

| Model | Conf Thresh | Precision | Recall | F1-Score | mAP@50 | Latency (ms) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock COCO YOLOv8n (Baseline) | 0.20 | 0.762 | 0.000 | 0.000 | 0.784 | 85.23 | 11.7 |
| Stock COCO YOLOv8n (Baseline) | 0.50 | 0.885 | 0.000 | 0.000 | 0.750 | 37.3 | 26.8 |
| Fine-Tuned YOLOv8n (Alcohol) | 0.20 | 0.914 | 0.932 | 0.923 | 0.941 | 43.24 | 23.1 |
| Fine-Tuned YOLOv8n (Alcohol) | 0.50 | 0.965 | 0.884 | 0.923 | 0.920 | 36.38 | 27.5 |