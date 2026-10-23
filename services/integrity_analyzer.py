"""Document-integrity analysis (Hallmark tier 5).

All checks here are *heuristics*. They produce evidence that needs
interpretation, never proof of forgery. Each finding records the rule that
produced it, and weights are deliberately small (see ``authenticity_scorer``).

PDF checks
  * white filled shapes drawn over text (manual-edit pattern)
  * text under such a shape that is invisible on the rendered page (covered text)
  * embedded text that is not visible in the rendered page (hidden text)
  * mixed fonts inside a single text line
  * editing software recorded in metadata, modification long after creation
  * multiple saved revisions in an unsigned file
Image checks
  * editing software recorded in EXIF
  * local noise inconsistency: background regions much cleaner than the
    surrounding paper noise (pasted digital content). Error-level analysis
    was evaluated and dropped because it did not separate genuine from
    spliced test images.
"""
from __future__ import annotations

import base64
import difflib
import io
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .evidence import Evidence, Finding, Status, Strength
from .security_analyzer import file_facts

EDITOR_TOOLS = re.compile(
    r"(?i)photoshop|gimp|paint\.net|pixlr|pdfescape|sejda|pdf-xchange\s*editor|smallpdf|ilovepdf|"
    r"pdffiller|dochub|pdf\s*editor|foxit\s*phantom|nitro\s*pro|inkscape|affinity\s*photo|snapseed|picsart")


# --------------------------------------------------------------------------- helpers

def _parse_pdf_date(s: str | None) -> datetime | None:
    if not s:
        return None
    m = re.match(r"D?:?(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?([Zz+\-])?(\d{2})?'?(\d{2})?", s)
    if not m:
        return None
    parts = [int(x) if x else d for x, d in zip(m.groups()[:6], (1, 1, 1, 0, 0, 0))]
    try:
        d = datetime(*parts)
    except ValueError:
        return None
    sign, hh, mm = m.group(7), m.group(8), m.group(9)
    if sign in ("+", "-") and hh:
        off = timedelta(hours=int(hh), minutes=int(mm or 0))
        d = (d - off) if sign == "+" else (d + off)
    return d.replace(tzinfo=timezone.utc)


def _words(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z]{4,}", text)]


def _overlap(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / area_b


def _heatmap_png(image: Image.Image, mask: np.ndarray) -> str:
    """Return a base64 PNG overlay highlighting ``mask`` on a downsized image."""
    base = image.convert("RGB").copy()
    base.thumbnail((900, 900))
    m = cv2.resize(mask.astype(np.uint8) * 255, base.size, interpolation=cv2.INTER_NEAREST)
    arr = np.array(base).astype(np.float32)
    overlay = arr.copy()
    overlay[m > 0] = overlay[m > 0] * 0.35 + np.array([214, 64, 69]) * 0.65
    out = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    b = io.BytesIO()
    out.save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


# --------------------------------------------------------------------------- PDF checks

def _pdf_white_overlays(page) -> list[Finding]:
    out = []
    if not page.width_pt:
        return out
    page_area = page.width_pt * page.height_pt
    for d in page.drawings:
        fill = d["fill"]
        if not fill or min(fill) < 0.95 or d.get("fill_opacity", 1.0) < 0.95:
            continue
        r = d["rect"]
        area = (r[2] - r[0]) * (r[3] - r[1])
        if area <= 20 or area > 0.25 * page_area:
            continue   # ignore hairlines and page/background panels
        covered = [s for s in page.text_spans if _overlap(r, s["bbox"]) > 0.5]
        if covered:
            out.append(Finding(
                "pdf_white_overlay",
                f"Page {page.number}: a small opaque white shape sits on a text line "
                f"(\"{covered[0]['text'][:40]}\"). Covering and retyping text is a common manual-edit pattern, "
                "though some templates use white panels legitimately.",
                -1, Strength.WEAK, page=page.number, data={"rect": [round(x, 1) for x in r]}))
    return out


def _pdf_covered_text(page, overlays: list[Finding], ocr_text: str, ocr_quality: str) -> list[Finding]:
    """Text under a white overlay that is NOT visible on the rendered page: corroborated cover-up."""
    if ocr_quality not in ("good", "fair") or not overlays:
        return []
    seen = set(_words(ocr_text))
    out = []
    for ov in overlays:
        r = ov.data["rect"]
        under = [s for s in page.text_spans if _overlap(r, s["bbox"]) > 0.5]
        hidden = [w for s in under for w in _words(s["text"])
                  if w not in seen and not difflib.get_close_matches(w, seen, n=1, cutoff=0.75)]
        if hidden:
            out.append(Finding(
                "pdf_covered_text",
                f"Page {page.number}: text in the PDF (\"{' '.join(hidden[:6])}\") lies under a white shape and is "
                "not visible on the rendered page — it appears to have been covered and replaced.",
                -1, Strength.MODERATE, page=page.number, data={"covered_words": hidden[:20], "rect": r}))
    return out


def _pdf_hidden_text(page, ocr_text: str, ocr_quality: str) -> list[Finding]:
    if ocr_quality not in ("good", "fair"):
        return []
    layer = _words(page.text_layer)
    if len(layer) < 5:
        return []
    seen = set(_words(ocr_text))
    hidden = []
    for w in dict.fromkeys(layer):
        if w in seen:
            continue
        if difflib.get_close_matches(w, seen, n=1, cutoff=0.75):
            continue
        hidden.append(w)
    ratio = len(hidden) / max(1, len(set(layer)))
    if len(hidden) >= 2 and ratio >= 0.10:
        strength = Strength.MODERATE if ratio >= 0.25 and len(hidden) >= 4 else Strength.WEAK
        return [Finding(
            "pdf_hidden_text",
            f"Page {page.number}: {len(hidden)} word(s) exist in the PDF's text layer but are not visible on the "
            f"rendered page (e.g. {', '.join(hidden[:5])}). Text may have been covered or replaced.",
            -1, strength, page=page.number, data={"hidden_words": hidden[:20], "ratio": round(ratio, 3)})]
    return []


def _pdf_mixed_fonts(page) -> list[Finding]:
    lines: dict[int, list[dict]] = {}
    for s in page.text_spans:
        key = int(round((s["bbox"][1] + s["bbox"][3]) / 2 / 3))
        lines.setdefault(key, []).append(s)
    odd = []
    for spans in lines.values():
        if len(spans) < 2:
            continue
        sizes = {s["size"] for s in spans}
        base = lambda f: re.sub(r"(?i)[-,](bold|italic|oblique|regular|medium|light).*$", "", f)
        fonts = {base(s["font"]) for s in spans}
        if len(fonts) > 1 and max(sizes) - min(sizes) <= 0.6:
            odd.append(" ".join(s["text"] for s in spans)[:60])
    if odd:
        return [Finding("pdf_mixed_fonts_in_line",
                        f"Page {page.number}: {len(odd)} line(s) mix different font families at the same size "
                        f"(e.g. \"{odd[0]}\"). Inserted text often uses a different font.",
                        -1, Strength.WEAK, page=page.number, data={"lines": odd[:5]})]
    return []


def _pdf_metadata(doc, facts) -> list[Finding]:
    out = []
    meta = doc.metadata
    tools = " ".join(str(meta.get(k, "")) for k in ("producer", "creator"))
    if EDITOR_TOOLS.search(tools):
        out.append(Finding("pdf_editor_software",
                           f"The PDF metadata names editing software ({tools.strip()[:80]}). This is common for edited "
                           "files but also for legitimately designed ones.", -1, Strength.WEAK))
    c, m = _parse_pdf_date(meta.get("creationDate")), _parse_pdf_date(meta.get("modDate"))
    now = datetime.now(timezone.utc)
    if c and c > now + timedelta(days=1):
        out.append(Finding("pdf_future_creation_date", "The PDF creation date is in the future.",
                           -1, Strength.WEAK, data={"creation": c.isoformat()}))
    if c and m and m - c > timedelta(days=1):
        out.append(Finding("pdf_modified_after_creation",
                           f"The PDF was modified {(m - c).days} day(s) after it was created.",
                           -1, Strength.WEAK, data={"creation": c.isoformat(), "modified": m.isoformat()}))
    if c and m and m < c - timedelta(days=1):   # tolerate time-zone bugs in PDF producers
        out.append(Finding("pdf_dates_inconsistent", "The PDF modification date is earlier than its creation date.",
                           -1, Strength.WEAK))
    return out


# --------------------------------------------------------------------------- image checks

def _flat_noise_anomalies(image: Image.Image, block: int = 24, z_thr: float = 5.0,
                          min_blocks: int = 6) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Find clusters of flat background blocks that are much *cleaner* than the rest.

    Scanned or photographed paper carries fairly uniform sensor/compression
    noise. Digitally rendered text or patches pasted into such an image are
    usually far cleaner than their surroundings. Only flat blocks (no text
    strokes) are compared, so text edges are never flagged on their own.

    Validated only on Hallmark's synthetic splice tests (see docs/EVALUATION.md);
    it cannot detect edits on noiseless, born-digital images.
    """
    g = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = g.shape
    bh, bw = h // block, w // block
    info: dict[str, Any] = {}
    if bh < 6 or bw < 6:
        return np.zeros((1, 1), bool), 0.0, {"applicable": False, "reason": "image too small"}
    hp = (g - cv2.GaussianBlur(g, (0, 0), 1.5))[: bh * block, : bw * block].reshape(bh, block, bw, block)
    gb = g[: bh * block, : bw * block].reshape(bh, block, bw, block)
    std = hp.std(axis=(1, 3))
    flat = (gb.max(axis=(1, 3)) - gb.min(axis=(1, 3))) < 30
    if flat.sum() < 50:
        return np.zeros_like(flat), 0.0, {"applicable": False, "reason": "too few flat background areas"}
    base_noise = float(np.median(std[flat]))
    info["background_noise"] = round(base_noise, 3)
    if base_noise < 0.6:
        return np.zeros_like(flat), 0.0, {"applicable": False, "background_noise": round(base_noise, 3),
                                          "reason": "background is noiseless (born-digital image)"}
    ls = np.log(std + 0.05)
    med = np.median(ls[flat])
    mad = np.median(np.abs(ls[flat] - med)) * 1.4826 + 1e-3
    z = np.zeros_like(ls)
    z[flat] = (ls[flat] - med) / mad
    mask = (z < -z_thr) & flat
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_blocks:
            keep |= lab == i
    info["applicable"] = True
    return keep, float(keep.mean()), info


def _image_checks(doc) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    details: dict[str, Any] = {}
    sw = " ".join(str(doc.exif.get(k, "")) for k in ("Software", "ProcessingSoftware", "HostComputer"))
    if EDITOR_TOOLS.search(sw):
        out.append(Finding("exif_editor_software",
                           f"The image metadata names editing software ({sw.strip()[:60]}).", -1, Strength.WEAK))
    img = doc.pages[0].image
    w, h = img.size
    if min(w, h) < 500:
        details["low_resolution"] = True
        return out, details      # too small for pixel forensics
    work = img.copy()
    work.thumbnail((2400, 2400))
    mask, frac, info = _flat_noise_anomalies(work)
    details["noise_check"] = info
    if frac > 0:
        out.append(Finding("noise_inconsistency",
                           f"A localised background region ({frac:.1%} of the image) is much cleaner than the paper "
                           "noise around it. Digitally pasted text or patches typically look like this.",
                           -1, Strength.WEAK, data={"fraction": round(frac, 4)}))
        details["heatmap_png_b64"] = _heatmap_png(work, mask)
    return out, details


# --------------------------------------------------------------------------- entry point

def analyse_integrity(doc, ocr_by_page: dict[int, Any], crypto_status: Status) -> Evidence:
    facts = file_facts(doc)
    findings: list[Finding] = []
    details: dict[str, Any] = {"sha256": facts["sha256"], "size_bytes": facts["size_bytes"]}
    limitations = [
        "Integrity checks are heuristics: they can miss skilled edits and can flag benign design choices.",
        "A SHA-256 fingerprint identifies this exact file but cannot prove authenticity without a trusted reference hash.",
    ]

    if doc.is_pdf:
        details["revisions"] = facts["revisions"]
        if facts["active_content"]:
            findings.append(Finding("pdf_active_content",
                                    "The PDF contains " + ", ".join(facts["active_content"]) +
                                    ". This is a security concern for the reader, not authenticity evidence.",
                                    0, Strength.NONE))
        if facts["revisions"] > 1 and crypto_status == Status.NOT_APPLICABLE:
            findings.append(Finding("pdf_incremental_updates",
                                    f"The unsigned PDF was saved {facts['revisions']} times (incremental updates), "
                                    "meaning content was added or changed after the first save.",
                                    -1, Strength.WEAK, data={"revisions": facts["revisions"]}))
        findings += _pdf_metadata(doc, facts)
        for page in doc.pages:
            overlays = _pdf_white_overlays(page)
            findings += overlays
            ocr = ocr_by_page.get(page.number)
            if ocr is not None:
                covered = _pdf_covered_text(page, overlays, ocr.text, ocr.quality)
                findings += covered
                if not covered:
                    findings += _pdf_hidden_text(page, ocr.text, ocr.quality)
            findings += _pdf_mixed_fonts(page)
        if not any(p.text_layer.strip() for p in doc.pages):
            limitations.append("The PDF has no text layer (likely a scan), so structure-level checks were limited.")
    else:
        f, d = _image_checks(doc)
        findings += f
        details.update(d)
        if d.get("low_resolution"):
            limitations.append("The image resolution is too low for pixel-level forensic checks.")
        nc = d.get("noise_check", {})
        if nc and not nc.get("applicable", True):
            limitations.append(f"The pixel-noise check could not run: {nc.get('reason')}.")

    negatives = [f for f in findings if f.direction < 0]
    if any(f.strength in (Strength.MODERATE, Strength.STRONG) for f in negatives):
        status = Status.FAIL
        expl = "Integrity checks found signs consistent with alteration."
    elif negatives:
        status = Status.INCONCLUSIVE
        expl = (f"{len(negatives)} weak integrity indicator(s) were found. Each has benign explanations, "
                "so they lower confidence rather than prove alteration.")
    elif crypto_status == Status.FAIL:
        status = Status.INCONCLUSIVE
        expl = ("The heuristic integrity checks found nothing further, but the signature check already shows "
                "the file changed after signing, so this result is not relied on.")
    else:
        status = Status.PASS
        expl = "No supported signs of alteration were found. This does not prove the document is genuine."
        findings.append(Finding("no_integrity_issues", expl, +1, Strength.WEAK))
    return Evidence("document_integrity", status, expl, findings=findings, limitations=limitations, details=details)
