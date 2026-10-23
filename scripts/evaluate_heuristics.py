"""Evaluate Hallmark's image/PDF integrity heuristics on controlled synthetic data.

This measures false-positive and detection rates of the heuristics on
synthetic certificates. It is NOT an estimate of real-world performance; it
exists so that heuristic thresholds are checked rather than guessed.

    python scripts/evaluate_heuristics.py  [--n 40]
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from services.document_reader import load_document          # noqa: E402
from services.integrity_analyzer import analyse_integrity    # noqa: E402
from services.ocr_service import run_ocr                    # noqa: E402
from services.evidence import Status                         # noqa: E402
from tests.fixtures import CERT_LINES, _font, certificate_image, certificate_pdf  # noqa: E402
from tests.make_synthetic_dataset import NAMES               # noqa: E402


def jpeg(img, q):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    return b.getvalue()


def noisy(img, sigma, seed):
    a = np.asarray(img).astype(np.float32) + np.random.default_rng(seed).normal(0, sigma, (img.height, img.width, 3))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def flagged(name, data, code):
    doc = load_document(name, data)
    ocr = {p.number: run_ocr(p.image) for p in doc.pages} if doc.is_pdf else {}
    ev = analyse_integrity(doc, ocr, Status.NOT_APPLICABLE)
    return any(f.code == code for f in ev.findings), ev


def main(n: int) -> dict:
    rnd = random.Random(7)
    res = {}
    for sigma in (2, 3, 5, 8):
        fp = tp = 0
        for i in range(n):
            lines = list(CERT_LINES)
            lines[3] = (rnd.choice(NAMES), 52, True)
            q = rnd.choice([75, 85, 92])
            genuine = Image.open(io.BytesIO(jpeg(noisy(certificate_image(lines, seed=i), sigma, i), q))).convert("RGB")
            fp += flagged("g.jpg", jpeg(genuine, q), "noise_inconsistency")[0]
            patch = Image.new("RGB", (rnd.randint(500, 760), 90), (250, 247, 240))
            ImageDraw.Draw(patch).text((20, 10), rnd.choice(NAMES), fill=(30, 20, 20), font=_font(52, True))
            spliced = genuine.copy()
            spliced.paste(patch, (rnd.randint(560, 700), rnd.randint(440, 480)))
            tp += flagged("s.jpg", jpeg(spliced, q), "noise_inconsistency")[0]
        res[f"noise_sigma_{sigma}"] = {"n": n, "false_positive_rate": fp / n, "detection_rate": tp / n}
        print(f"noise check, scan noise sigma={sigma}: FP {fp}/{n}, detected {tp}/{n}")
    # PDF white-box cover-up
    fp = tp = 0
    for i in range(min(n, 20)):
        lines = list(CERT_LINES)
        lines[3] = (rnd.choice(NAMES), 52, True)
        fp += flagged("c.pdf", certificate_pdf(lines), "pdf_covered_text")[0]
        tp += flagged("w.pdf", certificate_pdf(lines, white_box=True), "pdf_covered_text")[0]
    k = min(n, 20)
    res["pdf_covered_text"] = {"n": k, "false_positive_rate": fp / k, "detection_rate": tp / k}
    print(f"PDF covered-text check: FP {fp}/{k}, detected {tp}/{k}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs" / "heuristics_eval.json"))
    a = ap.parse_args()
    r = main(a.n)
    Path(a.out).write_text(json.dumps(r, indent=2))
    print("written", a.out)
