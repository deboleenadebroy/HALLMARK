"""OCR with light preprocessing (orientation, deskew, contrast, upscaling).

Runs fully offline using the local Tesseract binary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
except Exception:  # pragma: no cover - import guard
    pytesseract = None
    TesseractNotFoundError = Exception

import os as _os

if pytesseract is not None and _os.environ.get("HALLMARK_TESSERACT_CMD"):
    # e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe when Tesseract is not on PATH
    pytesseract.pytesseract.tesseract_cmd = _os.environ["HALLMARK_TESSERACT_CMD"]

MIN_SHORT_SIDE = 1200    # upscale small scans before OCR
MAX_LONG_SIDE = 3500     # downscale huge images to bound runtime


@dataclass
class OCRResult:
    text: str = ""
    mean_confidence: float = 0.0      # 0-100, over recognised words
    word_count: int = 0
    words: list[dict] = field(default_factory=list)   # {text, conf, bbox}
    rotation_applied: int = 0
    deskew_angle: float = 0.0
    available: bool = True
    error: str | None = None

    @property
    def quality(self) -> str:
        if not self.available:
            return "unavailable"
        if self.word_count < 5:
            return "unreadable"
        if self.mean_confidence >= 80:
            return "good"
        if self.mean_confidence >= 60:
            return "fair"
        return "poor"


def tesseract_available() -> bool:
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _resize(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    short, long_ = min(h, w), max(h, w)
    scale = 1.0
    if short < MIN_SHORT_SIDE:
        scale = MIN_SHORT_SIDE / short
    if long_ * scale > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_
    if abs(scale - 1.0) > 0.05:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    return gray


def _detect_rotation(gray: np.ndarray) -> int:
    """Use Tesseract OSD to find 0/90/180/270 rotation. Returns degrees clockwise to correct."""
    try:
        osd = pytesseract.image_to_osd(gray, config="--psm 0 -c min_characters_to_try=20")
        m = re.search(r"Rotate:\s+(\d+)", osd)
        c = re.search(r"Orientation confidence:\s+([\d.]+)", osd)
        if m and c and float(c.group(1)) >= 2.0:
            return int(m.group(1)) % 360
    except Exception:
        pass
    return 0


def _deskew_angle(gray: np.ndarray) -> float:
    """Estimate small skew (|angle| <= 15 degrees) from dark-pixel distribution."""
    inv = cv2.bitwise_not(gray)
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 500:
        return 0.0
    lines = cv2.HoughLinesP(thr, 1, np.pi / 720, threshold=200,
                            minLineLength=gray.shape[1] // 6, maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) <= 15:
            angles.append(a)
    if len(angles) < 3:
        return 0.0
    angle = float(np.median(angles))
    return angle if abs(angle) >= 0.3 else 0.0


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def preprocess(image: Image.Image) -> tuple[np.ndarray, int, float]:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = _resize(gray)
    rotation = _detect_rotation(gray)
    if rotation:
        k = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation]
        gray = cv2.rotate(gray, k)
    angle = _deskew_angle(gray)
    if angle:
        gray = _rotate(gray, angle)
    # Contrast normalisation for faded scans.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray, rotation, angle


def run_ocr(image: Image.Image) -> OCRResult:
    if not tesseract_available():
        return OCRResult(available=False, error="Tesseract OCR is not installed.")
    try:
        gray, rotation, angle = preprocess(image)
        data = pytesseract.image_to_data(gray, config="--oem 1 --psm 3",
                                         output_type=pytesseract.Output.DICT)
    except Exception as exc:
        return OCRResult(available=False, error=f"OCR failed: {exc.__class__.__name__}")

    words, confs = [], []
    lines: dict[tuple, list[str]] = {}
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if not txt or conf < 0:
            continue
        words.append({"text": txt, "conf": conf,
                      "bbox": (data["left"][i], data["top"][i], data["width"][i], data["height"][i])})
        if re.search(r"[A-Za-z0-9]", txt):
            confs.append(conf)
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(txt)
    text = "\n".join(" ".join(v) for _, v in sorted(lines.items()))
    return OCRResult(text=text, mean_confidence=float(np.mean(confs)) if confs else 0.0,
                     word_count=len(confs), words=words, rotation_applied=rotation,
                     deskew_angle=round(angle, 2))
