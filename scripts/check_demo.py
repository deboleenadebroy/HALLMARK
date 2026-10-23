"""Run every demo file through the real Hallmark pipeline and print the live results.

    python scripts/check_demo.py

Nothing here is pre-computed: each score is calculated now, exactly as the web app does it.
The 'expected' column is what the file is designed to show; a mismatch is reported, not hidden.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from services.ocr_service import tesseract_available  # noqa: E402
from services.pipeline import analyse_bytes            # noqa: E402

EXPECTED = {
    "1_signed_certificate.pdf": "strong_authentic",
    "2_altered_after_signing.pdf": "likely_forged",
    "3_content_added_after_signing.pdf": "likely_forged",
    "4_name_covered_and_retyped.pdf": "suspicious",
    "5_clean_scan.jpg": "inconclusive",
    "6_name_pasted_into_scan.jpg": "inconclusive",
    "7_future_dated_certificate.pdf": "suspicious",
    "8_not_a_certificate.pdf": "unavailable",
    "kaggle_genuine_gradecard.jpg": "inconclusive",
    "kaggle_forged_logo_seal_signature.jpg": "inconclusive",
}


def main() -> int:
    demo = ROOT / "demo"
    if not demo.exists():
        print("No demo/ folder. Create it with:  python scripts/make_demo_files.py --trust")
        return 1
    if not tesseract_available():
        print("Tesseract OCR is not available. Install it (see DEMO.md) or set HALLMARK_TESSERACT_CMD.")
        return 1
    files = sorted(p for p in demo.rglob("*") if p.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg"))
    problems = 0
    print(f"{'file':40s} {'score':>6s}  {'result':34s} check")
    for p in files:
        r = analyse_bytes(p.name, p.read_bytes())
        s = r["score"]["score"]
        band = r["score"]["band"]
        exp = EXPECTED.get(p.name)
        ok = exp is None or band["key"] == exp
        problems += not ok
        mark = "ok" if ok else f"UNEXPECTED (designed to show: {exp})"
        print(f"{p.name:40s} {('%5.1f%%' % s) if s is not None else '   —  '}  {band['label']:34s} {mark}")
    if not (ROOT / "trust_anchors" / "demo_root_ca.pem").exists():
        print("\nNote: trust_anchors/demo_root_ca.pem is missing, so file 1 cannot show as Verified.")
    print("\nAll demo files behave as designed." if not problems else f"\n{problems} file(s) differ from the design.")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
