"""
Alcohol Bottle Detection Engine
Uses YOLOv8 to detect bottles in video frames.
Supports both webcam and uploaded video file via socket commands.
Run: python detector.py
"""

import cv2
import json
import time
import base64
import threading
import logging
import numpy as np
import socketio
import sys
import subprocess
from pathlib import Path
from ultralytics import YOLO
from ocr_reader import OCRReader
import difflib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DETECTOR] %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
SERVER_URL        = "http://localhost:3001"
CAMERA_INDEX      = 0
CONFIDENCE_THRESH = 0.20
DEBOUNCE_FRAMES   = 3
FRAME_SKIP        = 1
MODEL_PATH        = Path(__file__).parent / "yolov8n_alcohol.pt"
MODEL_NAME        = str(MODEL_PATH) if MODEL_PATH.exists() else "yolov8n.pt"
BRAND_DB_PATH     = Path(__file__).parent.parent / "brand-db" / "alcohol_brands.json"
BOTTLE_CLASSES    = {"bottle", "wine glass", "cup", "alcohol"}

# ── Brand DB ────────────────────────────────────────────────────────────────
def load_brand_db(path: Path) -> set:
    with open(path) as f:
        data = json.load(f)
    brands = {b.lower().strip() for b in data["brands"]}
    log.info(f"Loaded {len(brands)} alcohol brand keywords")
    return brands

def match_brand(label_text: str, brand_db: set, threshold=0.7) -> bool:
    label_lower = label_text.lower().strip()
    if not label_lower:
        return False
    # 1. Direct match check
    for brand in brand_db:
        if brand in label_lower:
            return True
    # 2. Fuzzy match word check (edit distance / ratio)
    words = label_lower.split()
    for brand in brand_db:
        for w in words:
            if len(w) >= 3 and len(brand) >= 3:
                ratio = difflib.SequenceMatcher(None, brand, w).ratio()
                if ratio >= threshold:
                    return True
    return False

# ── Pre-check ───────────────────────────────────────────────────────────────
class PreChecker:
    def __init__(self):
        self.prev_gray = None

    def reset(self):
        self.prev_gray = None

    def check(self, frame: np.ndarray) -> dict:
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        # Guard against size mismatch (e.g. source resolution changed)
        if self.prev_gray is not None and self.prev_gray.shape != gray.shape:
            self.prev_gray = None
        motion   = float(np.mean(cv2.absdiff(gray, self.prev_gray))) if self.prev_gray is not None else 0.0
        self.prev_gray = gray
        return {
            "ok":           mean_lum >= 15,
            "occluded":     mean_lum < 15,
            "luminance":    round(mean_lum, 1),
            "motion_score": round(motion, 2),
        }

# ── Debouncer ───────────────────────────────────────────────────────────────
class Debouncer:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self.count     = 0
        self.alerted   = False

    def update(self, detected: bool) -> tuple:
        newly_alerted = False
        if detected:
            self.count += 1
            if self.count >= self.threshold and not self.alerted:
                self.alerted = True
                newly_alerted = True
        else:
            self.count   = 0
            self.alerted = False
        return newly_alerted, self.alerted

# ── Detection Engine ────────────────────────────────────────────────────────
class DetectionEngine:
    def __init__(self):
        self.brand_db  = load_brand_db(BRAND_DB_PATH)
        self.ocr       = OCRReader()
        self.model     = YOLO(MODEL_NAME)
        self.pre       = PreChecker()
        self.debouncer = Debouncer(DEBOUNCE_FRAMES)
        self.sio       = socketio.Client()
        self.connected = False

        # Video switching state
        self._lock           = threading.Lock()
        self._pending_video  = None   # path string if a video was requested
        self._use_screen     = False  # flag to switch to screen capture
        self._stop_current   = False  # signal current capture to stop
        self._popup_process  = None   # track the popup process
        self._last_alert_time = 0     # track when the last detection happened

        self._setup_socket()

    def _setup_socket(self):
        sio = self.sio

        @sio.event
        def connect():
            self.connected = True
            log.info("Connected to Node.js server")

        @sio.event
        def disconnect():
            self.connected = False
            log.warning("Disconnected from Node.js server")

        @sio.on("process_video")
        def on_process_video(data):
            log.info(f"Server requested video: {data['path']}")
            with self._lock:
                self._pending_video = data["path"]
                self._stop_current  = True

        @sio.on("use_screen")
        def on_use_screen():
            log.info("Server requested switch to screen capture")
            with self._lock:
                self._use_screen   = True
                self._stop_current = True

    def connect_server(self):
        try:
            self.sio.connect(SERVER_URL, headers={"client": "python"},
                             auth=None, transports=["websocket"])
            # Pass client type as query param
            self.sio.disconnect()
            self.sio.connect(SERVER_URL + "?client=python", transports=["websocket"])
        except Exception as e:
            log.error(f"Cannot connect to server: {e}")

    def emit_alert(self, payload: dict):
        if self.connected:
            self.sio.emit("bottle_detected", payload)

    def emit_frame(self, frame: np.ndarray, detections: list):
        if not self.connected:
            return
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        b64 = base64.b64encode(buf).decode("utf-8")
        self.sio.emit("frame_update", {
            "image":      b64,
            "detections": detections,
            "timestamp":  time.time(),
        })

    def process_frame(self, frame: np.ndarray) -> dict:
        pre = self.pre.check(frame)
        if pre["occluded"]:
            if self.connected:
                self.sio.emit("camera_alert", {"reason": "occluded", "luminance": pre["luminance"]})
            return {"status": "occluded"}

        results      = self.model(frame, verbose=False, conf=CONFIDENCE_THRESH)[0]
        detections   = []
        bottle_found = False
        best_det     = None

        # Extract persons for action detection
        persons = [list(map(int, box.xyxy[0])) for box in results.boxes if self.model.names[int(box.cls[0])].lower() == "person"]

        def check_action(bx1, by1, bx2, by2, person_list):
            for px1, py1, px2, py2 in person_list:
                head_bottom = py1 + (py2 - py1) * 0.4
                intersect_x = max(0, min(bx2, px2) - max(bx1, px1))
                intersect_y = max(0, min(by2, head_bottom) - max(by1, py1))
                if intersect_x > 0 and intersect_y > 0:
                    return True
            return False

        # Process upright results
        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.model.names[cls_id].lower()
            conf     = float(box.conf[0])
            if cls_name not in BOTTLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            drinking = check_action(x1, y1, x2, y2, persons)
            
            ocr_text = ""
            if conf >= CONFIDENCE_THRESH and cls_name in BOTTLE_CLASSES:
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                ocr_text = self.ocr.read(crop)

            brand_text = ocr_text if ocr_text else cls_name
            brand_hit  = match_brand(brand_text, self.brand_db)
            color      = (0, 0, 255) if drinking else ((0, 200, 0) if brand_hit else (200, 200, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            display_label = f"{brand_text[:15]} {conf:.2f}" if ocr_text else f"{cls_name} {conf:.2f}"
            label = display_label + (" [DRINKING]" if drinking else "")
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            det = {"class": cls_name, "confidence": round(conf, 3),
                   "bbox": [x1, y1, x2, y2], "brand_matched": brand_hit,
                   "ocr_text": ocr_text,
                   "drinking_action": drinking,
                   "timestamp": time.time()}
            detections.append(det)

            if conf >= CONFIDENCE_THRESH:
                bottle_found = True
                if not best_det or conf > best_det["confidence"] or (drinking and not best_det.get("drinking_action", False)):
                    best_det = det

        # If no contraband found upright, try upside down
        if not bottle_found:
            frame_180 = cv2.rotate(frame, cv2.ROTATE_180)
            results_180 = self.model(frame_180, verbose=False, conf=CONFIDENCE_THRESH)[0]
            persons_180 = [list(map(int, box.xyxy[0])) for box in results_180.boxes if self.model.names[int(box.cls[0])].lower() == "person"]
            
            for box in results_180.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.model.names[cls_id].lower()
                conf     = float(box.conf[0])
                if cls_name not in BOTTLE_CLASSES:
                    continue

                # Coordinate adjustment
                h, w = frame.shape[:2]
                rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                nx1, ny1 = w - rx2, h - ry2
                nx2, ny2 = w - rx1, h - ry1
                
                drinking = check_action(rx1, ry1, rx2, ry2, persons_180)
                
                ocr_text = ""
                if conf >= CONFIDENCE_THRESH and cls_name in BOTTLE_CLASSES:
                    crop = frame_180[max(0, ry1):min(h, ry2), max(0, rx1):min(w, rx2)]
                    ocr_text = self.ocr.read(crop)

                brand_text = ocr_text if ocr_text else cls_name
                brand_hit  = match_brand(brand_text, self.brand_db)
                color      = (0, 0, 255) if drinking else ((0, 200, 0) if brand_hit else (200, 200, 0))
                cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), color, 2)
                
                display_label = f"{brand_text[:15]} {conf:.2f} (inv)" if ocr_text else f"{cls_name} {conf:.2f} (inv)"
                label = display_label + (" [DRINKING]" if drinking else "")
                cv2.putText(frame, label, (nx1, ny1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                det = {"class": f"{cls_name} (inv)", "confidence": round(conf, 3),
                       "bbox": [nx1, ny1, nx2, ny2], "brand_matched": brand_hit,
                       "ocr_text": ocr_text,
                       "drinking_action": drinking,
                       "timestamp": time.time()}
                detections.append(det)

                if conf >= CONFIDENCE_THRESH:
                    bottle_found = True
                    if not best_det or conf > best_det["confidence"] or (drinking and not best_det.get("drinking_action", False)):
                        best_det = det

        newly_alerted, is_alerting = self.debouncer.update(bottle_found)
        if newly_alerted and best_det:
            self.emit_alert(best_det)
            # Launch the OS-level popup overlay if not already running
            if self._popup_process is None:
                popup_path = Path(__file__).parent / "popup.py"
                mode = "drinking" if best_det.get("drinking_action") else "bottle"
                self._popup_process = subprocess.Popen([sys.executable, str(popup_path), mode])
                
        if is_alerting:
            self._last_alert_time = time.time()
            
        if not is_alerting and self._popup_process is not None:
            # Hold the popup open for at least 4 seconds after the bottle disappears
            if time.time() - self._last_alert_time > 4.0:
                self._popup_process.terminate()
                self._popup_process = None

        self.emit_frame(frame, detections)
        return {"status": "ok", "detections": detections}

    def run_capture(self, source, label="webcam"):
        """Run detection on a given cv2 capture source or screen."""
        # Reset pre-checker so stale prev_gray from another source can't cause size mismatches
        self.pre.reset()

        if source == "screen":
            from mss import MSS
            with MSS() as sct:
                monitor = sct.monitors[1]  # primary monitor
                log.info("Opened screen capture")
                frame_idx = 0
                while True:
                    with self._lock:
                        if self._stop_current:
                            self._stop_current = False
                            log.info("Stopping screen capture")
                            break
                    
                    # Capture screen
                    img = sct.grab(monitor)
                    frame = np.array(img)
                    # Convert BGRA to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    frame_idx += 1
                    if frame_idx % FRAME_SKIP != 0:
                        continue

                    self.process_frame(frame)
                    
                    # Throttle screen capture to ~20 FPS to save CPU
                    time.sleep(0.05)
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            log.error(f"Cannot open source: {source}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(source, str) else -1
        log.info(f"Opened {label} — {total_frames if total_frames > 0 else 'live'} frames")

        frame_idx = 0
        while True:
            # Check for stop/switch signal
            with self._lock:
                if self._stop_current:
                    self._stop_current = False
                    log.info(f"Stopping {label}")
                    break

            ret, frame = cap.read()
            if not ret:
                log.info(f"End of {label}")
                if isinstance(source, str) and self.connected:
                    self.sio.emit("video_done", {"path": source, "filename": label})
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            self.process_frame(frame)

            # Send video progress every 30 frames
            if isinstance(source, str) and total_frames > 0 and frame_idx % 30 == 0:
                pct = round((frame_idx / total_frames) * 100)
                if self.connected:
                    self.sio.emit("video_progress", {"percent": pct, "frame": frame_idx})

            # Optional local preview
            cv2.imshow("Alcohol Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

    def run(self):
        self.connect_server()
        current_source = "screen"

        try:
            while True:
                self.run_capture(current_source,
                                 label=current_source)

                # Decide what to open next
                with self._lock:
                    if self._pending_video:
                        current_source       = self._pending_video
                        self._pending_video  = None
                    elif self._use_screen:
                        current_source       = "screen"
                        self._use_screen     = False
                    else:
                        # No switch requested — loop back to same source
                        if current_source == "screen":
                            time.sleep(0.5)
                        else:
                            break
        finally:
            cv2.destroyAllWindows()
            if self.connected:
                self.sio.disconnect()
            log.info("Detection engine stopped")


if __name__ == "__main__":
    engine = DetectionEngine()
    engine.run()
