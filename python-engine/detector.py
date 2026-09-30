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
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DETECTOR] %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
SERVER_URL        = "http://localhost:3001"
CAMERA_INDEX      = 0
CONFIDENCE_THRESH = 0.45  # Stricter threshold to eliminate low-confidence hallucinations
DEBOUNCE_FRAMES   = 3
FRAME_SKIP        = 1
COCO_MODEL_PATH   = Path(__file__).parent / "yolov8n.pt"
ALC_MODEL_PATH    = Path(__file__).parent / "yolov8n_alcohol.pt"
BRAND_DB_PATH     = Path(__file__).parent.parent / "brand-db" / "alcohol_brands.json"
BOTTLE_CLASSES    = {"bottle", "wine glass", "cup"}

NON_ALCOHOL_KEYWORDS = {
    "non-alcoholic", "non alcoholic", "alcohol free", "alcohol-free",
    "0.0%", "zero alcohol", "rubbing alcohol", "sanitizer", "hand sanitizer",
    "mineral water", "water", "juice", "apple juice", "orange juice"
}

# ── Brand DB ────────────────────────────────────────────────────────────────
def load_brand_db(path: Path) -> set:
    with open(path) as f:
        data = json.load(f)
    GENERIC_WORDS = {"alcohol", "liquor", "spirit", "brew"}
    brands = {b.lower().strip() for b in data["brands"] if b.lower().strip() not in GENERIC_WORDS}
    log.info(f"Loaded {len(brands)} alcohol brand keywords")
    return brands

def match_brand(label_text: str, brand_db: set, threshold=0.85) -> tuple:
    """
    Evaluates text for genuine alcohol brand names.
    Uses whole-word boundaries and negative keyword filtering to avoid false positives.
    """
    text = label_text.lower().strip()
    if not text:
        return False, ""

    # Negative phrase suppression (e.g. alcohol-free, rubbing alcohol, water)
    for neg in NON_ALCOHOL_KEYWORDS:
        if neg in text:
            return False, ""

    # 1. Exact word boundary match
    for brand in brand_db:
        pattern = r'\b' + re.escape(brand) + r'\b'
        if re.search(pattern, text):
            return True, brand

    # 2. Strict fuzzy match only on words with length >= 5
    words = re.findall(r'[a-z0-9]+', text)
    for brand in brand_db:
        if len(brand) < 5:
            continue
        for w in words:
            if len(w) >= 4 and abs(len(w) - len(brand)) <= 2:
                if difflib.SequenceMatcher(None, brand, w).ratio() >= threshold:
                    return True, brand

    return False, ""

# ── Geometry & Banner Filter ─────────────────────────────────────────────────
def is_valid_bottle_geometry(box: list, frame_shape: tuple) -> tuple:
    """
    Validates physical container geometry to filter out horizontal web banners,
    advertisement strips, full-screen windows, and tiny pixel noise.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    fh, fw = frame_shape[:2]

    # Reject tiny noise or thin vertical strips (scrollbars, minimaps, gutters)
    if w < 32 or h < 55:
        return False, "too_small_or_thin"

    # Aspect ratio: real bottles/glasses have 0.65 <= h/w <= 5.0
    ar = h / max(1, w)
    if ar < 0.65:
        return False, f"horizontal_banner_ar_{ar:.2f}"
    if ar > 5.0:
        return False, f"vertical_minimap_scrollbar_ar_{ar:.2f}"

    # Reject wide horizontal banner ads spanning more than 60% of screen width
    if w > 0.60 * fw and h < 0.40 * fh:
        return False, "horizontal_screen_banner"

    # Reject full-screen bounding boxes
    if w > 0.85 * fw and h > 0.85 * fh:
        return False, "full_screen_box"

    return True, "ok"

# ── Pre-check ───────────────────────────────────────────────────────────────
class PreChecker:
    def __init__(self):
        self.prev_gray = None

    def reset(self):
        self.prev_gray = None

    def check(self, frame: np.ndarray) -> dict:
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
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

        # Primary physical container detector (COCO pretrained: bottle, wine glass, cup, person)
        coco_file = COCO_MODEL_PATH if COCO_MODEL_PATH.exists() else "yolov8n.pt"
        self.model_coco = YOLO(str(coco_file))

        # Fine-tuned alcohol vs non-alcohol visual classifier
        alc_file = ALC_MODEL_PATH if ALC_MODEL_PATH.exists() else coco_file
        self.model_alcohol = YOLO(str(alc_file))

        # Alias for backwards compatibility
        self.model = self.model_alcohol

        self.pre       = PreChecker()
        self.debouncer = Debouncer(DEBOUNCE_FRAMES)
        self.sio       = socketio.Client()
        self.connected = False

        # Video switching state
        self._lock           = threading.Lock()
        self._pending_video  = None
        self._use_screen     = False
        self._stop_current   = False
        self._popup_process  = None
        self._last_alert_time = 0

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
        if frame is None or frame.size == 0:
            return {"status": "empty", "detections": []}
        pre = self.pre.check(frame)
        if pre["occluded"]:
            if self.connected:
                self.sio.emit("camera_alert", {"reason": "occluded", "luminance": pre["luminance"]})
            return {"status": "occluded", "detections": []}

        h, w = frame.shape[:2]
        detections   = []
        bottle_found = False
        best_det     = None

        # ── 1. PRIMARY REQUIREMENT: PHYSICAL BOTTLE / CONTAINER PRESENCE ─────
        # Physical objects (bottle, wine glass, cup, person) detected via COCO model.
        coco_res = self.model_coco(frame, verbose=False, conf=CONFIDENCE_THRESH)[0]

        # Extract persons for drinking action detection
        persons = [
            list(map(int, box.xyxy[0]))
            for box in coco_res.boxes
            if int(box.cls[0]) == 0  # 0: person in COCO
        ]

        def check_action(bx1, by1, bx2, by2, person_list):
            for px1, py1, px2, py2 in person_list:
                head_bottom = py1 + (py2 - py1) * 0.45
                intersect_x = max(0, min(bx2, px2) - max(bx1, px1))
                intersect_y = max(0, min(by2, head_bottom) - max(by1, py1))
                if intersect_x > 0 and intersect_y > 0:
                    return True
            return False

        # Gather candidate containers and enforce geometry validation
        candidate_containers = []
        for box in coco_res.boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.model_coco.names[cls_id].lower()
            conf     = float(box.conf[0])

            if cls_name not in BOTTLE_CLASSES:
                continue

            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            valid_geo, reason = is_valid_bottle_geometry([bx1, by1, bx2, by2], frame.shape)
            if not valid_geo:
                log.debug(f"Banner/non-bottle candidate rejected ({reason}): [{bx1},{by1},{bx2},{by2}]")
                continue

            candidate_containers.append((cls_name, conf, [bx1, by1, bx2, by2]))

        # If no physical container is present, DO NOT flag anything.
        # Banners, website headlines, articles mentioning 'alcohol' are immediately dismissed.
        if not candidate_containers:
            self.debouncer.update(False)
            self.emit_frame(frame, detections)
            return {"status": "clear", "detections": []}

        # ── 2. ALCOHOL VERIFICATION & OPTIONAL TEXT MATCHING ─────────────────
        for c_name, c_conf, (x1, y1, x2, y2) in candidate_containers:
            drinking = check_action(x1, y1, x2, y2, persons)
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue

            # 2a. Visual fine-tuned model classification on container crop
            alc_conf = 0.0
            no_alc_conf = 0.0
            if self.model_alcohol is not self.model_coco:
                r_alc = self.model_alcohol(crop, verbose=False, conf=0.30)[0]
                for b in r_alc.boxes:
                    name = self.model_alcohol.names[int(b.cls[0])]
                    conf_val = float(b.conf[0])
                    if name == "alcohol":
                        alc_conf = max(alc_conf, conf_val)
                    elif name == "no-alcohol":
                        no_alc_conf = max(no_alc_conf, conf_val)
            else:
                alc_conf = c_conf

            # 2b. Optional Text / Brand matching (OCR on container crop only)
            ocr_text = ""
            brand_hit = False
            matched_brand_name = ""
            if self.ocr.available:
                ocr_text = self.ocr.read(crop)
                if ocr_text:
                    brand_hit, matched_brand_name = match_brand(ocr_text, self.brand_db)

            # Determine whether this container is alcohol
            is_alcohol = False
            if brand_hit:
                is_alcohol = True
            elif alc_conf >= 0.40 and alc_conf >= no_alc_conf:
                is_alcohol = True
            elif c_name in {"wine glass"}:
                is_alcohol = True

            # Suppress non-alcohol beverages when clearly classified as no-alcohol and no brand match
            if no_alc_conf > 0.65 and not brand_hit:
                is_alcohol = False

            if not is_alcohol:
                # Regular non-alcohol container (water bottle, soda, etc.)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
                cv2.putText(frame, f"{c_name} (clear)", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                continue

            bottle_found = True
            disp_conf = max(alc_conf, c_conf)
            label_name = matched_brand_name.title() if brand_hit else f"Alcohol {c_name.title()}"
            if drinking:
                label_name += " [DRINKING]"

            color = (0, 0, 255) if drinking else ((0, 200, 0) if brand_hit else (0, 140, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_name} {disp_conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            det = {
                "class": label_name,
                "confidence": round(disp_conf, 3),
                "bbox": [x1, y1, x2, y2],
                "brand_matched": brand_hit,
                "brand_name": matched_brand_name,
                "ocr_text": ocr_text,
                "drinking_action": drinking,
                "timestamp": time.time(),
            }
            detections.append(det)

            if not best_det or disp_conf > best_det["confidence"] or (drinking and not best_det.get("drinking_action", False)):
                best_det = det

        newly_alerted, is_alerting = self.debouncer.update(bottle_found)
        if newly_alerted and best_det:
            self.emit_alert(best_det)
            if self._popup_process is None:
                popup_path = Path(__file__).parent / "popup.py"
                mode = "drinking" if best_det.get("drinking_action") else "bottle"
                self._popup_process = subprocess.Popen([sys.executable, str(popup_path), mode])

        if is_alerting:
            self._last_alert_time = time.time()

        if not is_alerting and self._popup_process is not None:
            if time.time() - self._last_alert_time > 4.0:
                try:
                    self._popup_process.terminate()
                except Exception:
                    pass
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
