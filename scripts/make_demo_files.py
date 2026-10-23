"""Generate a demo set of synthetic certificates that show each Hallmark outcome.

    python scripts/make_demo_files.py            # writes demo/
    python scripts/make_demo_files.py --trust    # also installs the demo CA into trust_anchors/

All documents and keys are synthetic and for demonstration only.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives import serialization
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from tests.fixtures import (CERT_LINES, _font, certificate_image, certificate_pdf, image_bytes,  # noqa: E402
                            modify_after_signing, sign_pdf, tamper_signed_bytes, test_pki)


def scan(img, sigma=3, seed=0):
    a = np.asarray(img).astype(np.float32) + np.random.default_rng(seed).normal(0, sigma, (img.height, img.width, 3))
    b = io.BytesIO()
    Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(b, "JPEG", quality=85)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def main(trust: bool) -> None:
    out = ROOT / "demo"
    out.mkdir(exist_ok=True)
    pdf = certificate_pdf()
    signed = sign_pdf(pdf)
    files = {
        "1_signed_certificate.pdf": signed,
        "2_altered_after_signing.pdf": tamper_signed_bytes(signed),
        "3_content_added_after_signing.pdf": modify_after_signing(signed),
        "4_name_covered_and_retyped.pdf": certificate_pdf(white_box=True),
        "5_clean_scan.jpg": image_bytes(scan(certificate_image()), "JPEG", 85),
        "7_future_dated_certificate.pdf": certificate_pdf(lines=list(CERT_LINES[:6]) + [
            ("Date of issue: 14 March 2031", 30, False), ("Certificate ID: NIT-2031-00731", 28, False)]),
        "8_not_a_certificate.pdf": certificate_pdf(lines=[
            ("Quarterly sales summary", 50, True), ("Revenue grew in the northern region", 32, False),
            ("Prepared for the board meeting", 32, False)]),
    }
    spliced = scan(certificate_image())
    patch = Image.new("RGB", (700, 90), (250, 247, 240))
    ImageDraw.Draw(patch).text((20, 10), "Rahul Verma", fill=(30, 20, 20), font=_font(52, True))
    spliced.paste(patch, (650, 460))
    b = io.BytesIO()
    spliced.save(b, "JPEG", quality=85)
    files["6_name_pasted_into_scan.jpg"] = b.getvalue()
    for name, data in sorted(files.items()):
        (out / name).write_bytes(data)
    _, ca, _, _ = test_pki()
    ca_pem = ca.public_bytes(serialization.Encoding.PEM)
    (out / "demo_root_ca.pem").write_bytes(ca_pem)
    if trust:
        (ROOT / "trust_anchors" / "demo_root_ca.pem").write_bytes(ca_pem)
    print(f"wrote {len(files)} demo files to {out}" + (" and installed the demo CA" if trust else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trust", action="store_true", help="copy the demo CA into trust_anchors/")
    main(ap.parse_args().trust)
