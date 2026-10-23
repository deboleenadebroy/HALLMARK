import io
import unittest
import warnings

from pypdf import PdfReader, PdfWriter

from services.document_reader import MAX_FILE_BYTES, DocumentError, load_document, validate_upload
from tests.fixtures import certificate_image, certificate_pdf, image_bytes

warnings.filterwarnings("ignore")


class UploadValidation(unittest.TestCase):
    def assertCode(self, code, fn, *a):
        with self.assertRaises(DocumentError) as cm:
            fn(*a)
        self.assertEqual(cm.exception.code, code)

    def test_accepts_supported_types(self):
        img = certificate_image()
        self.assertEqual(validate_upload("a.png", image_bytes(img)), "png")
        self.assertEqual(validate_upload("a.JPG", image_bytes(img, "JPEG")), "jpeg")
        self.assertEqual(validate_upload("a.jpeg", image_bytes(img, "JPEG")), "jpeg")
        self.assertEqual(validate_upload("a.pdf", certificate_pdf()), "pdf")

    def test_rejects_unsupported_extension(self):
        self.assertCode("unsupported_type", validate_upload, "a.gif", b"GIF89a....")
        self.assertCode("unsupported_type", validate_upload, "notes.txt", b"hello")
        self.assertCode("unsupported_type", validate_upload, "noext", b"%PDF-1.4")

    def test_rejects_empty_and_large(self):
        self.assertCode("empty_file", validate_upload, "a.pdf", b"")
        self.assertCode("too_large", validate_upload, "a.pdf", b"%PDF-" + b"0" * MAX_FILE_BYTES)

    def test_rejects_content_mismatch(self):
        self.assertCode("extension_mismatch", validate_upload, "a.png", certificate_pdf())
        self.assertCode("unrecognised_content", validate_upload, "a.pdf", b"this is not a pdf at all")

    def test_corrupted_files(self):
        png = image_bytes(certificate_image())
        self.assertCode("corrupted", load_document, "a.png", png[:200])
        pdf = certificate_pdf()
        self.assertCode("corrupted", load_document, "a.pdf", pdf[:60])

    def test_encrypted_pdf(self):
        w = PdfWriter(clone_from=PdfReader(io.BytesIO(certificate_pdf())))
        w.encrypt("secret")
        b = io.BytesIO()
        w.write(b)
        self.assertCode("encrypted", load_document, "a.pdf", b.getvalue())

    def test_multipage_pdf_pages_loaded(self):
        doc = load_document("m.pdf", certificate_pdf(pages=3))
        self.assertEqual(doc.total_pages, 3)
        self.assertEqual([p.number for p in doc.pages], [1, 2, 3])
        self.assertTrue(all(p.text_layer.strip() for p in doc.pages))
        self.assertTrue(all(p.image.size[0] > 1000 for p in doc.pages))

    def test_image_exif_orientation_applied(self):
        doc = load_document("c.png", image_bytes(certificate_image()))
        self.assertEqual(doc.pages[0].image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
