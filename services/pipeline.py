"""Orchestrates a full Hallmark analysis for one uploaded file."""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from models.predict import analyse_visual

from .authenticity_scorer import compute_score
from .certificate_verifier import analyse_cryptography
from .content_analyzer import analyse_identity, analyse_issuer, analyse_registry, looks_like_certificate
from .document_reader import DocumentError, load_document
from .evidence import Evidence, Status
from .integrity_analyzer import analyse_integrity
from .ocr_service import OCRResult, run_ocr
from .tier6_classifier import classify

for name in ("pyhanko", "pyhanko_certvalidator", "pyhanko.sign"):
    logging.getLogger(name).setLevel(logging.CRITICAL)

MAX_OCR_PAGES = 10
MIN_LAYER_WORDS = 15


def _text_quality(ocrs: dict[int, OCRResult], layer_used: dict[int, bool]) -> str:
    if layer_used and all(layer_used.values()):
        return "good"
    qs = [o.quality for n, o in ocrs.items() if not layer_used.get(n)]
    if not qs:
        return "unavailable"
    order = ["good", "fair", "poor", "unreadable", "unavailable"]
    # document quality = median page quality (robust to one blank page)
    qs.sort(key=order.index)
    return qs[len(qs) // 2]


def analyse_bytes(filename: str, data: bytes) -> dict[str, Any]:
    t0 = time.time()
    doc = load_document(filename, data)          # raises DocumentError

    ocrs: dict[int, OCRResult] = {}
    page_texts: dict[int, str] = {}
    layer_used: dict[int, bool] = {}
    notes: list[str] = []
    for p in doc.pages:
        layer_words = len(p.text_layer.split())
        if p.number <= MAX_OCR_PAGES:
            ocrs[p.number] = run_ocr(p.image)
        if layer_words >= MIN_LAYER_WORDS:
            page_texts[p.number] = p.text_layer
            layer_used[p.number] = True
        else:
            page_texts[p.number] = ocrs[p.number].text if p.number in ocrs else ""
            layer_used[p.number] = False
    if len(doc.pages) > MAX_OCR_PAGES:
        notes.append(f"OCR was limited to the first {MAX_OCR_PAGES} pages.")
    if doc.truncated:
        notes.append(f"Only the first {len(doc.pages)} of {doc.total_pages} pages were analysed.")
    if any(not o.available for o in ocrs.values()):
        notes.append("OCR was unavailable: " + next(o.error for o in ocrs.values() if not o.available))

    quality = _text_quality(ocrs, layer_used)
    full_text = "\n".join(page_texts.values())
    in_scope = looks_like_certificate(full_text)

    crypto = analyse_cryptography(doc)
    integrity = analyse_integrity(doc, ocrs, crypto.status)
    # Where the text layer contains words that are not visible, read what a person actually sees.
    for f in integrity.findings:
        if f.code in ("pdf_covered_text", "pdf_hidden_text") and f.page in ocrs and layer_used.get(f.page):
            page_texts[f.page] = ocrs[f.page].text
            layer_used[f.page] = False
            notes.append(f"Page {f.page}: details were read from the rendered page, because its text layer "
                         "contains text that is not visible.")
    quality = _text_quality(ocrs, layer_used)
    identity, parsed = analyse_identity(page_texts, quality)
    visual = analyse_visual(doc.pages)
    evidence: list[Evidence] = [analyse_registry(), crypto, integrity, analyse_issuer(parsed), identity, visual]

    readable = quality not in ("unreadable", "unavailable") or crypto.status in (Status.PASS, Status.FAIL) \
        or visual.status in (Status.PASS, Status.FAIL)
    score = compute_score(evidence, in_scope=in_scope, readable=readable)
    if not in_scope and score.score is not None:
        notes.append("The document does not clearly look like a certificate; treat the score with extra caution.")
    assessment = classify(evidence, score)

    return {
        "id": uuid.uuid4().hex,
        "filename": filename,
        "analysed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "duration_s": round(time.time() - t0, 2),
        "file": {"kind": doc.kind, "pages_total": doc.total_pages, "pages_analysed": len(doc.pages),
                 "size_bytes": len(data), "sha256": integrity.details.get("sha256"),
                 "metadata": {k: str(v)[:120] for k, v in doc.metadata.items()}},
        "text": {"quality": quality, "source": "text layer" if layer_used and all(layer_used.values())
                 else ("OCR" if not any(layer_used.values()) else "text layer + OCR"),
                 "ocr_confidence": round(sum(o.mean_confidence for o in ocrs.values()) / len(ocrs), 1) if ocrs else None,
                 "in_scope": in_scope},
        "extracted": parsed.to_dict(),
        "score": score.to_dict(),
        "assessment": assessment,
        "evidence": [e.to_dict() for e in evidence],
        "notes": notes,
        "limitations": [
            "Hallmark gives an evidence-based estimate, not a legally recognised verification.",
            "The score is a heuristic combination of checks; it is not a statistically validated probability.",
            "Only the issuing institution can confirm that a certificate was genuinely issued.",
        ],
    }


__all__ = ["analyse_bytes", "DocumentError"]
