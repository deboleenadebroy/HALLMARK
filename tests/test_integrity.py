import io
import unittest
import warnings

import numpy as np
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter

from services.document_reader import load_document
from services.evidence import Status
from services.integrity_analyzer import analyse_integrity
from services.ocr_service import run_ocr, tesseract_available
from tests.fixtures import _font, certificate_image, certificate_pdf, image_bytes

warnings.filterwarnings("ignore")


def noisy_jpeg(img, sigma=3, q=85, seed=0):
    a = np.asarray(img).astype(np.float32) + np.random.default_rng(seed).normal(0, sigma, (img.height, img.width, 3))
    b = io.BytesIO()
    Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def splice(img):
    patch = Image.new("RGB", (700, 90), (250, 247, 240))
    ImageDraw.Draw(patch).text((20, 10), "Rahul Verma", fill=(30, 20, 20), font=_font(52, True))
    out = img.copy()
    out.paste(patch, (650, 460))
    return out


def run(name, data, crypto=Status.NOT_APPLICABLE):
    doc = load_document(name, data)
    ocr = {p.number: run_ocr(p.image) for p in doc.pages} if tesseract_available() else {}
    return analyse_integrity(doc, ocr, crypto)


class ImageIntegrity(unittest.TestCase):
    def test_clean_scans_not_flagged(self):
        for seed in range(4):
            ev = run("c.jpg", image_bytes(noisy_jpeg(certificate_image(seed=seed), seed=seed), "JPEG", 85))
            self.assertNotIn("noise_inconsistency", [f.code for f in ev.findings], seed)

    def test_spliced_patch_flagged(self):
        for seed in range(4):
            img = splice(noisy_jpeg(certificate_image(seed=seed), seed=seed))
            ev = run("s.jpg", image_bytes(img, "JPEG", 85))
            self.assertIn("noise_inconsistency", [f.code for f in ev.findings], seed)
            self.assertIn("heatmap_png_b64", ev.details)
            self.assertNotEqual(ev.status, Status.PASS)

    def test_born_digital_image_check_not_applicable(self):
        ev = run("c.png", image_bytes(certificate_image()))
        self.assertFalse(ev.details["noise_check"]["applicable"])
        self.assertTrue(any("pixel-noise check could not run" in l for l in ev.limitations))

    def test_low_resolution_skips_pixel_checks(self):
        ev = run("s.png", image_bytes(certificate_image().resize((400, 280))))
        self.assertTrue(ev.details.get("low_resolution"))


class PDFIntegrity(unittest.TestCase):
    def test_clean_pdf_passes(self):
        ev = run("c.pdf", certificate_pdf())
        self.assertEqual(ev.status, Status.PASS)

    def test_white_overlay_detected(self):
        ev = run("w.pdf", certificate_pdf(white_box=True))
        codes = [f.code for f in ev.findings]
        self.assertIn("pdf_white_overlay", codes)
        self.assertNotEqual(ev.status, Status.PASS)

    @unittest.skipUnless(tesseract_available(), "Tesseract not installed")
    def test_covered_text_detected(self):
        ev = run("w.pdf", certificate_pdf(white_box=True))
        covered = [f for f in ev.findings if f.code == "pdf_covered_text"]
        self.assertTrue(covered)   # the original name is in the text layer but hidden under the white box
        self.assertIn("priya", covered[0].data["covered_words"])
        self.assertEqual(ev.status, Status.FAIL)

    @unittest.skipUnless(tesseract_available(), "Tesseract not installed")
    def test_clean_pdf_has_no_hidden_text(self):
        ev = run("c.pdf", certificate_pdf())
        self.assertFalse([f for f in ev.findings if f.code in ("pdf_hidden_text", "pdf_covered_text")])

    def test_incremental_update_on_unsigned_pdf(self):
        w = PdfWriter(clone_from=PdfReader(io.BytesIO(certificate_pdf())))
        b = io.BytesIO()
        w.write(b)
        data = b.getvalue() + b"\n% appended revision\n%%EOF\n"
        ev = run("u.pdf", data)
        self.assertIn("pdf_incremental_updates", [f.code for f in ev.findings])

    def test_signature_failure_suppresses_clean_pass(self):
        ev = run("c.pdf", certificate_pdf(), crypto=Status.FAIL)
        self.assertEqual(ev.status, Status.INCONCLUSIVE)
        self.assertNotIn("no_integrity_issues", [f.code for f in ev.findings])

    def test_sha256_recorded(self):
        data = certificate_pdf()
        import hashlib
        self.assertEqual(run("c.pdf", data).details["sha256"], hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
