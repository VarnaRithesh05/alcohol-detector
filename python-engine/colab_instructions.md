# Google Colab Training Guide

This guide describes how to run the fine-tuning of the YOLOv8 alcohol model using Google Colab T4 GPU acceleration, which completes in a few minutes instead of hours on a local CPU.

---

### Step 1: Zip File Location
We have pre-packed your current local dataset into a zip file.
- Location: [dataset.zip](file:///d:/alcohol-detector/python-engine/dataset.zip)

---

### Step 2: Open Colab & Configure GPU
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Create a new notebook.
3. In the menu, go to **Runtime > Change runtime type**.
4. Under *Hardware accelerator*, select **T4 GPU** and click **Save**.

---

### Step 3: Install Ultralytics
In the first cell, install the Ultralytics library:
```python
!pip install ultralytics
```

---

### Step 4: Upload the Dataset Zip
1. On the left sidebar of Colab, click the **Files (folder)** icon.
2. Drag and drop [dataset.zip](file:///d:/alcohol-detector/python-engine/dataset.zip) from your local computer into the files list.
3. Wait for the upload circle to finish.

---

### Step 5: Unzip the Dataset
In a new cell, run the following to extract your dataset:
```python
!unzip -q dataset.zip -d /content
```

---

### Step 6: Fix data.yaml paths
To make sure YOLO can find the files in Google Colab's directories, recreate the `data.yaml` config file:
```python
import yaml

data_yaml = {
    "path": "/content/dataset",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {
        0: "alcohol",
        1: "no-alcohol"
    }
}

with open("/content/dataset/data.yaml", "w") as f:
    yaml.dump(data_yaml, f)
```

---

### Step 7: Train the Model
Run the 40-epoch fine-tuning using T4 GPU acceleration:
```python
from ultralytics import YOLO

# Load baseline model
model = YOLO('yolov8n.pt')

# Fine-tune model
results = model.train(
    data='/content/dataset/data.yaml',
    epochs=40,
    imgsz=640,
    batch=16,
    device=0,
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    optimizer='auto',
    project='runs',
    name='alcohol_real_training_v2'
)
```

*Note: While running, keep an eye on the training logs. If `box_loss`/`cls_loss` are still decreasing and validation `mAP50` is climbing at Epoch 40, you can run another cell training for 20 more epochs by loading `model = YOLO('/content/runs/alcohol_real_training_v2/weights/best.pt')`.*

---

### Step 8: Download Best Weights
1. In the file explorer sidebar, navigate to `/content/runs/detect/alcohol_real_training_v2/weights/` (or check under `/content/runs/alcohol_real_training_v2/weights/`).
2. Right-click on **`best.pt`** and choose **Download**.
3. Rename the downloaded file to **`yolov8n_alcohol.pt`**.
4. Replace the old file in your local directory at: `d:\alcohol-detector\python-engine\yolov8n_alcohol.pt`.
5. Once you replace the file, let me know, and I will run the expanded evaluation!
