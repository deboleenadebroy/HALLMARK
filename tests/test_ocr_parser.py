import unittest
import warnings
from datetime import date

from services.certificate_parser import check_consistency, extract_ids, parse_certificate
from services.content_analyzer import analyse_identity
from services.evidence import Status
from services.ocr_service import run_ocr, tesseract_available
from tests.fixtures import CERT_LINES, certificate_image, low_quality_scan

warnings.filterwarnings("ignore")

SIMPLILEARN = """CERTIFICATE OF COMPLETION
simplilearn
This is to certify that
Yug Agrawal
has successfully completed the course
Basics of Data Structures and Algorithms
on 31st Aug 2025
Certificate code: 8886355
CERTIFICATE ificate IFICATE CERT IFICATE
www.simplilearn.com"""


class IdentifierExtraction(unittest.TestCase):
    def test_simplilearn_regression_no_fragment_ids(self):
        ids = [c.value for c in parse_certificate(SIMPLILEARN).accepted_ids]
        self.assertEqual(ids, ["8886355"])
        for frag in ("IFICATE", "ificate"):
            self.assertNotIn(frag, ids)

    def test_simplilearn_no_identifier_conflict(self):
        issues = check_consistency(parse_certificate(SIMPLILEARN), today=date(2026, 1, 1))
        self.assertNotIn("conflicting_identifier", [i.code for i in issues])

    def test_fragments_after_label_rejected(self):
        cands = extract_ids("Certificate No: IFICATE")
        self.assertTrue(cands and not cands[0].accepted)
        self.assertIn("fragment", cands[0].reason)

    def test_legitimate_alphanumeric_ids_preserved(self):
        for text, val in [("Credential ID: UC-8f2a91c3-77d1", "UC-8f2a91c3-77d1"),
                          ("Serial No. 2024/AB/00731", "2024/AB/00731"),
                          ("Certificate Number - NIT-2024-00731", "NIT-2024-00731"),
                          ("Registration No: 21070126045", "21070126045"),
                          ("Verification code: XKQZTRPW", "XKQZTRPW")]:
            acc = [c.value for c in extract_ids(text) if c.accepted]
            self.assertEqual(acc, [val], text)

    def test_real_conflict_same_field_detected(self):
        pc = parse_certificate("Certificate ID: ABC-1234\nfooter text\nCertificate ID: ABC-9999")
        codes = [i.code for i in check_consistency(pc)]
        self.assertIn("conflicting_identifier", codes)

    def test_different_fields_not_compared(self):
        pc = parse_certificate("Certificate ID: ABC-1234\nRoll No: 556677\nPhone 9876543210 2024")
        self.assertNotIn("conflicting_identifier", [i.code for i in check_consistency(pc)])

    def test_unlabelled_numbers_not_ids(self):
        pc = parse_certificate("Awarded 2024. Scored 98 in 12 modules. Ref 4 pages.")
        self.assertEqual(pc.accepted_ids, [])


class Scope(unittest.TestCase):
    def test_academic_records_in_scope(self):
        from services.content_analyzer import looks_like_certificate
        self.assertTrue(looks_like_certificate("GRADE CARD\nVISVESVARAYA TECHNOLOGICAL UNIVERSITY"))
        self.assertTrue(looks_like_certificate("Statement of Marks / Mark Sheet - Semester III"))
        self.assertFalse(looks_like_certificate("Quarterly sales figures for the northern region"))


class FieldsAndConsistency(unittest.TestCase):
    def test_fields_extracted(self):
        pc = parse_certificate(SIMPLILEARN)
        self.assertEqual(pc.recipient, "Yug Agrawal")
        self.assertEqual(pc.course, "Basics of Data Structures and Algorithms")
        self.assertEqual(pc.dates[0].value, date(2025, 8, 31))
        self.assertIn("simplilearn.com", pc.domains)

    def test_future_and_expiry_contradictions(self):
        pc = parse_certificate("Date of issue: 10 May 2030\nValid until 1 January 2029")
        codes = {i.code for i in check_consistency(pc, today=date(2026, 10, 1))}
        self.assertIn("future_issue_date", codes)
        self.assertIn("expiry_before_issue", codes)

    def test_placeholder_detected(self):
        pc = parse_certificate("This is to certify that\nYOUR NAME HERE\nhas completed Lorem Ipsum")
        self.assertIn("placeholder_text", [i.code for i in check_consistency(pc)])

    def test_missing_id_is_not_failure(self):
        text = "\n".join(t for t, *_ in CERT_LINES if "ID" not in t)
        ev, pc = analyse_identity({1: text}, "good")
        self.assertEqual(pc.accepted_ids, [])
        self.assertNotEqual(ev.status, Status.FAIL)
        self.assertEqual(ev.status, Status.PASS)

    def test_unreadable_text_is_inconclusive_not_fail(self):
        ev, _ = analyse_identity({1: ""}, "unreadable")
        self.assertEqual(ev.status, Status.INCONCLUSIVE)


@unittest.skipUnless(tesseract_available(), "Tesseract not installed")
class OCR(unittest.TestCase):
    def test_clean_image(self):
        r = run_ocr(certificate_image())
        self.assertEqual(r.quality, "good")
        self.assertIn("Priya Raman", r.text)
        self.assertIn("NIT-2024-00731", r.text)

    def test_rotated_image_corrected(self):
        r = run_ocr(certificate_image().rotate(90, expand=True))
        self.assertIn(r.rotation_applied, (90, 270))
        self.assertIn("Priya", r.text)

    def test_skewed_image(self):
        r = run_ocr(certificate_image().rotate(4, expand=True, fillcolor=(255, 255, 255)))
        self.assertIn("Priya", r.text)

    def test_low_quality_scan_reported(self):
        r = run_ocr(low_quality_scan(certificate_image()))
        self.assertIn(r.quality, ("unreadable", "poor", "fair"))
        self.assertNotEqual(r.quality, "good")


if __name__ == "__main__":
    unittest.main()
