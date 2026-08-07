"""
ocr_reader.py — Optional EasyOCR label text extraction
Install: pip install easyocr

Drop-in replacement for the class-name-only brand matching in detector.py.
Usage:
    from ocr_reader import OCRReader
    reader = OCRReader()
    text = reader.read(cropped_bottle_image)
    matched = match_brand(text, brand_db)
"""

import numpy as np
import logging

log = logging.getLogger(__name__)

class OCRReader:
    def __init__(self):
        try:
            import easyocr
            self.reader = easyocr.Reader(["en"], gpu=True, verbose=False)
            self.available = True
            log.info("EasyOCR initialised (GPU)")
        except ImportError:
            self.reader = None
            self.available = False
            log.warning("easyocr not installed — falling back to class-name matching. "
                        "Run: pip install easyocr")

    def read(self, crop: np.ndarray) -> str:
        """
        Extract text from a cropped bottle image.
        Returns joined text string, lowercased.
        Falls back to empty string if OCR unavailable or fails.
        """
        if not self.available or crop is None or crop.size == 0:
            return ""
        try:
            results = self.reader.readtext(crop, detail=0, paragraph=True)
            text = " ".join(results).lower().strip()
            if text:
                log.debug(f"OCR read: '{text}'")
            return text
        except Exception as e:
            log.warning(f"OCR failed: {e}")
            return ""


# ── How to integrate into detector.py ──────────────────────────────────────
#
# 1. At top of detector.py, add:
#       from ocr_reader import OCRReader
#
# 2. In DetectionEngine.__init__, add:
#       self.ocr = OCRReader()
#
# 3. In process_frame(), after extracting bbox, add:
#       x1, y1, x2, y2 = map(int, box.xyxy[0])
#       crop = frame[y1:y2, x1:x2]
#       ocr_text = self.ocr.read(crop)
#       brand_text = ocr_text if ocr_text else cls_name
#       brand_hit = match_brand(brand_text, self.brand_db)
#
# That's it — the DB lookup works identically on OCR text or class names.
