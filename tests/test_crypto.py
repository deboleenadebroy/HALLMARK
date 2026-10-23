import unittest
import warnings
from unittest import mock

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

from services import certificate_verifier as cv
from services.document_reader import load_document
from services.evidence import Status
from services.security_analyzer import sha256_hex
from tests.fixtures import (anchor_dir_with_test_ca, certificate_image, certificate_pdf, image_bytes,
                            modify_after_signing, sign_pdf, tamper_signed_bytes)

warnings.filterwarnings("ignore")
DATA = b"Hallmark test payload"


class RSAVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.pub = cls.key.public_key()

    def test_valid_pkcs1v15(self):
        sig = self.key.sign(DATA, padding.PKCS1v15(), hashes.SHA256())
        self.assertTrue(cv.verify_rsa_signature(self.pub, DATA, sig))

    def test_valid_pss_pem_key(self):
        sig = self.key.sign(DATA, padding.PSS(padding.MGF1(hashes.SHA384()), padding.PSS.MAX_LENGTH), hashes.SHA384())
        pem = self.pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        self.assertTrue(cv.verify_rsa_signature(pem, DATA, sig, "sha384", "pss"))

    def test_invalid_signature(self):
        sig = bytearray(self.key.sign(DATA, padding.PKCS1v15(), hashes.SHA256()))
        sig[10] ^= 0xFF
        self.assertFalse(cv.verify_rsa_signature(self.pub, DATA, bytes(sig)))
        good = self.key.sign(DATA, padding.PKCS1v15(), hashes.SHA256())
        self.assertFalse(cv.verify_rsa_signature(self.pub, DATA + b"x", good))

    def test_wrong_padding_fails(self):
        sig = self.key.sign(DATA, padding.PKCS1v15(), hashes.SHA256())
        self.assertFalse(cv.verify_rsa_signature(self.pub, DATA, sig, padding="pss"))

    def test_unsupported_algorithms(self):
        sig = self.key.sign(DATA, padding.PKCS1v15(), hashes.SHA256())
        for h in ("md5", "sha1"):
            with self.assertRaises(cv.UnsupportedAlgorithm):
                cv.verify_rsa_signature(self.pub, DATA, sig, h)
        with self.assertRaises(cv.UnsupportedAlgorithm):
            cv.verify_rsa_signature(self.pub, DATA, sig, padding="oaep")

    def test_short_and_non_rsa_keys_rejected(self):
        small = rsa.generate_private_key(public_exponent=65537, key_size=1024)
        sig = small.sign(DATA, padding.PKCS1v15(), hashes.SHA256())
        with self.assertRaises(cv.UnsupportedAlgorithm):
            cv.verify_rsa_signature(small.public_key(), DATA, sig)
        eck = ec.generate_private_key(ec.SECP256R1()).public_key()
        with self.assertRaises(cv.UnsupportedAlgorithm):
            cv.verify_rsa_signature(eck, DATA, b"x")

    def test_sha256(self):
        self.assertEqual(sha256_hex(b"abc"),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


class PDFSignatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf = certificate_pdf()
        cls.signed = sign_pdf(cls.pdf)
        cls.anchors = anchor_dir_with_test_ca()

    def evidence(self, data, anchors=None, name="c.pdf"):
        with mock.patch.object(cv, "TRUST_ANCHOR_DIR", anchors or cv.TRUST_ANCHOR_DIR.parent / "nonexistent"):
            return cv.analyse_cryptography(load_document(name, data))

    def codes(self, ev):
        return [f.code for f in ev.findings]

    def test_unsigned_pdf_not_applicable(self):
        ev = self.evidence(self.pdf)
        self.assertEqual(ev.status, Status.NOT_APPLICABLE)

    def test_image_not_applicable(self):
        ev = self.evidence(image_bytes(certificate_image()), name="c.png")
        self.assertEqual(ev.status, Status.NOT_APPLICABLE)

    def test_valid_signature_missing_trust_anchor(self):
        ev = self.evidence(self.signed)
        self.assertEqual(ev.status, Status.PASS)
        self.assertEqual(self.codes(ev), ["signature_valid_untrusted"])
        self.assertFalse(ev.details["trust_anchors_configured"])
        sig = ev.details["signatures"][0]
        self.assertEqual(sig["algorithm"], "sha256_rsa")
        self.assertEqual(sig["key_bits"], 2048)
        self.assertTrue(any("trust anchors" in l for l in ev.limitations))

    def test_valid_trusted_signature(self):
        ev = self.evidence(self.signed, self.anchors)
        self.assertEqual(ev.status, Status.PASS)
        self.assertEqual(self.codes(ev), ["signature_valid_trusted"])

    def test_altered_signed_content(self):
        ev = self.evidence(tamper_signed_bytes(self.signed), self.anchors)
        self.assertEqual(ev.status, Status.FAIL)
        self.assertIn("signed_content_altered", self.codes(ev))

    def test_modified_after_signing(self):
        ev = self.evidence(modify_after_signing(self.signed), self.anchors)
        self.assertEqual(ev.status, Status.FAIL)
        self.assertIn("modified_after_signing", self.codes(ev))

    def test_weak_digest_not_relied_on(self):
        try:
            weak = sign_pdf(self.pdf, md_algorithm="sha1")
        except Exception:
            self.skipTest("pyHanko refuses to create SHA-1 signatures")
        ev = self.evidence(weak, self.anchors)
        self.assertNotEqual(ev.status, Status.PASS)

    def test_unsupported_signature_reported(self):
        fake = cv.SignatureReport(field_name="Sig", unsupported="NotImplementedError: algorithm not supported")
        with mock.patch.object(cv, "inspect_pdf_signatures", return_value=[fake]):
            ev = cv.analyse_cryptography(load_document("c.pdf", self.pdf))
        self.assertEqual(ev.status, Status.INCONCLUSIVE)
        self.assertIn("signature_unsupported", self.codes(ev))


if __name__ == "__main__":
    unittest.main()
