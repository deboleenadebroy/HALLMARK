import io
import os
import re
import tempfile
import unittest
import warnings
from unittest import mock

from models import predict
from services import certificate_verifier as cv
from services.pipeline import analyse_bytes
from services.report_generator import render_report
from tests.fixtures import (CERT_LINES, anchor_dir_with_test_ca, certificate_image, certificate_pdf, image_bytes,
                            sign_pdf, tamper_signed_bytes)

warnings.filterwarnings("ignore")


def setUpModule():
    # Integration tests run without a visual model unless one is installed deliberately.
    global _env
    _env = mock.patch.dict(os.environ, {"HALLMARK_MODEL_DIR": tempfile.mkdtemp()})
    _env.start()
    predict.reset_cache()


def tearDownModule():
    _env.stop()
    predict.reset_cache()


class Pipeline(unittest.TestCase):
    def test_png_image(self):
        r = analyse_bytes("c.png", image_bytes(certificate_image()))
        self.assertEqual(r["extracted"]["recipient"], "Priya Raman")
        self.assertIsNotNone(r["score"]["score"])
        self.assertEqual(r["score"]["band"]["key"], "inconclusive")
        cats = {e["category"]: e["status"] for e in r["evidence"]}
        self.assertEqual(cats["cryptography"], "NOT_APPLICABLE")
        self.assertEqual(cats["visual_model"], "NOT_PERFORMED")
        self.assertEqual(cats["registry"], "NOT_PERFORMED")

    def test_jpeg_image(self):
        r = analyse_bytes("c.jpg", image_bytes(certificate_image(), "JPEG", 90))
        self.assertIsNotNone(r["score"]["score"])

    def test_multipage_pdf(self):
        r = analyse_bytes("m.pdf", certificate_pdf(pages=3))
        self.assertEqual(r["file"]["pages_analysed"], 3)
        self.assertEqual(r["text"]["source"], "text layer")

    def test_signed_trusted_pdf_verified(self):
        with mock.patch.object(cv, "TRUST_ANCHOR_DIR", anchor_dir_with_test_ca()):
            r = analyse_bytes("s.pdf", sign_pdf(certificate_pdf()))
        self.assertEqual(r["assessment"]["classification"], "VERIFIED")
        self.assertGreaterEqual(r["score"]["score"], 90)

    def test_tampered_signed_pdf(self):
        r = analyse_bytes("t.pdf", tamper_signed_bytes(sign_pdf(certificate_pdf())))
        self.assertLess(r["score"]["score"], 20)
        self.assertNotIn("no signs of alteration", r["assessment"]["explanation"])

    def test_unreadable_scan_withholds_score(self):
        blank = certificate_image(lines=[]).resize((900, 630))
        r = analyse_bytes("b.png", image_bytes(blank))
        self.assertIsNone(r["score"]["score"])
        self.assertEqual(r["assessment"]["classification"], "INCONCLUSIVE")

    def test_non_certificate_out_of_scope(self):
        lines = [("Quarterly sales figures", 50, True), ("Revenue grew in the northern region", 32, False),
                 ("Prepared for the board meeting", 32, False), ("Please see the attached tables", 32, False)]
        r = analyse_bytes("x.pdf", certificate_pdf(lines=lines))
        self.assertFalse(r["text"]["in_scope"])
        self.assertIsNone(r["score"]["score"])

    def test_conflicting_evidence_flagged(self):
        lines = list(CERT_LINES) + [("Date of issue: 2 February 2031", 30, False)]
        with mock.patch.object(cv, "TRUST_ANCHOR_DIR", anchor_dir_with_test_ca()):
            r = analyse_bytes("c.pdf", sign_pdf(certificate_pdf(lines=lines)))
        self.assertTrue(r["score"]["conflict"])
        self.assertNotEqual(r["assessment"]["classification"], "VERIFIED")

    def test_covered_name_reads_visible_text(self):
        r = analyse_bytes("w.pdf", certificate_pdf(white_box=True))
        self.assertEqual(r["extracted"]["recipient"], "Rahul Verma")
        self.assertEqual(r["score"]["band"]["key"], "suspicious")

    def test_score_independent_of_certificate_id(self):
        with_id = analyse_bytes("a.pdf", certificate_pdf())
        no_id = analyse_bytes("b.pdf", certificate_pdf(lines=[l for l in CERT_LINES if "ID" not in l[0]]))
        self.assertEqual(with_id["score"]["score"], no_id["score"]["score"])

    def test_report_contents(self):
        r = analyse_bytes("c.png", image_bytes(certificate_image()))
        html = render_report(r)
        for s in ("Hallmark", "Estimated Authenticity Score", "c.png", "Visual model findings",
                  "Cryptographic results", "Document integrity", "Known limitations", "Recommendations",
                  "not a legally", r["file"]["sha256"]):
            self.assertIn(s, html)
        self.assertNotRegex(html, r'href="https?://')     # no external verification URLs


class WebApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import create_app
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def post(self, path, name, data):
        return self.client.post(path, data={"certificate": (io.BytesIO(data), name)},
                                content_type="multipart/form-data")

    def test_index(self):
        r = self.client.get("/")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("Your mark of authenticity.", html)
        self.assertIn("Analyse Certificate", html)
        self.assertEqual(html.count('type="file"'), 1)          # exactly one upload box
        for forbidden in ("Certificate ID</label>", "verification URL", "public key"):
            self.assertNotIn(forbidden, html)

    def test_analyse_and_download(self):
        r = self.post("/analyse", "c.png", image_bytes(certificate_image()))
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("Estimated Authenticity Score", html)
        self.assertIn("Download Report", html)
        rid = re.search(r'/report/([0-9a-f]{32})', html).group(1)
        rep = self.client.get(f"/report/{rid}")
        self.assertEqual(rep.status_code, 200)
        self.assertIn("attachment", rep.headers["Content-Disposition"])
        self.assertIn("Estimated Authenticity Score", rep.get_data(as_text=True))

    def test_api(self):
        r = self.post("/api/analyse", "c.pdf", certificate_pdf())
        j = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertIn("score", j)
        self.assertEqual(len(j["evidence"]), 6)

    def test_unsupported_and_corrupted(self):
        r = self.post("/analyse", "a.gif", b"GIF89a")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Unsupported file type", r.get_data(as_text=True))
        r = self.post("/api/analyse", "a.pdf", b"%PDF-1.4 broken")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"], "corrupted")

    def test_too_large(self):
        r = self.post("/analyse", "a.pdf", b"%PDF-" + b"0" * (21 * 1024 * 1024))
        self.assertEqual(r.status_code, 413)

    def test_expired_report(self):
        self.assertEqual(self.client.get("/report/" + "0" * 32).status_code, 404)
        self.assertEqual(self.client.get("/report/../../etc").status_code, 404)


if __name__ == "__main__":
    unittest.main()
