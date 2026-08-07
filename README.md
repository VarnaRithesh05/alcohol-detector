# Alcohol Bottle Detector

Real-time alcohol bottle detection system using YOLOv8 + Node.js + React.

## Architecture

```
[Webcam] → [Python/YOLOv8] → [Node.js/Socket.io] → [React Dashboard]
```

- Python engine detects bottles via YOLOv8, matches against alcohol brand DB
- Node.js backend relays events via WebSocket
- React dashboard shows live feed + instant alert popup + event log

## Prerequisites

- Python 3.9+ (https://python.org)
- Node.js 18+  (https://nodejs.org)
- A webcam

GPU (optional but recommended):
- NVIDIA GPU with CUDA 11.8+ for fast inference
- Without GPU: runs on CPU at ~8fps (still functional)

## Quick Start — open 3 terminals

**Terminal 1 — Backend**
```
start_backend.bat
```

**Terminal 2 — Detection Engine**
```
start_detector.bat
```
First run downloads YOLOv8s weights (~22MB) automatically.

**Terminal 3 — Dashboard**
```
start_frontend.bat
```
Opens http://localhost:3000 in your browser.

## Configuration (python-engine/detector.py)

| Setting | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | 0 | Webcam index, or path to video file |
| `CONFIDENCE_THRESH` | 0.75 | Min detection confidence (0–1) |
| `DEBOUNCE_FRAMES` | 3 | Consecutive frames before alert fires |
| `MODEL_NAME` | yolov8s.pt | nano=fastest, small=balanced, large=accurate |

## Tuning for accuracy vs speed

| Model | Speed (GPU) | Speed (CPU) | Accuracy |
|---|---|---|---|
| yolov8n.pt | ~4ms | ~60ms | Good |
| yolov8s.pt | ~12ms | ~120ms | Better |
| yolov8m.pt | ~25ms | ~250ms | Best |

## Adding more alcohol brands

Edit `brand-db/alcohol_brands.json` and add to the `brands` array.
The detector hot-reloads this file on restart.

## Optional: Real OCR label reading

Install easyocr for actual text recognition from labels:
```
pip install easyocr
```
Then uncomment the easyocr section in `python-engine/detector.py`.

## Project Structure

```
alcohol-detector/
├── backend/
│   ├── server.js          # Node.js + Express + Socket.io
│   └── package.json
├── python-engine/
│   ├── detector.py        # YOLOv8 detection loop
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   ├── useDetectorSocket.js
│   │   └── components/
│   │       ├── StatusBar.js
│   │       ├── CameraFeed.js
│   │       ├── AlertPanel.js
│   │       ├── StatsPanel.js
│   │       └── AlertToast.js
│   └── package.json
├── brand-db/
│   └── alcohol_brands.json
├── start_backend.bat
├── start_detector.bat
└── start_frontend.bat
```
