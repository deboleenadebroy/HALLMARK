"""Controlled test documents and throwaway test keys.

Nothing here uses production keys: a fresh test CA and signer are generated
per test session. Certificates rendered here are synthetic.
"""
from __future__ import annotations

import datetime as dt
import io
import random
import tempfile
from functools import lru_cache
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "/usr/share/fonts/dejavu", "/usr/share/fonts"]


def _font(size: int, bold: bool = False):
    names = (["DejaVuSerif-Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSerif-Bold.ttf"] if bold
             else ["DejaVuSerif.ttf", "DejaVuSans.ttf", "LiberationSerif-Regular.ttf"])
    for d in FONT_DIRS:
        for n in names:
            p = Path(d) / n
            if p.exists():
                return ImageFont.truetype(str(p), size)
        for p in Path(d).rglob("*.ttf") if Path(d).exists() else []:
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


CERT_LINES = [
    ("Northbridge Institute of Technology", 44, True),
    ("CERTIFICATE OF COMPLETION", 56, True),
    ("This is to certify that", 32, False),
    ("Priya Raman", 52, True),
    ("has successfully completed the course", 32, False),
    ("Applied Machine Learning Foundations", 38, True),
    ("Date of issue: 14 March 2024", 30, False),
    ("Certificate ID: NIT-2024-00731", 28, False),
]


def certificate_image(lines=CERT_LINES, size=(2000, 1400), seed=0) -> Image.Image:
    rnd = random.Random(seed)
    img = Image.new("RGB", size, (250, 247, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, size[0] - 40, size[1] - 40], outline=(86, 28, 36), width=10)
    y = 160
    for text, fs, bold in lines:
        f = _font(fs, bold)
        w = d.textlength(text, font=f)
        d.text(((size[0] - w) / 2, y), text, fill=(30, 20, 20), font=f)
        y += int(fs * 2.0)
    d.line([size[0] * 0.6, size[1] - 200, size[0] * 0.85, size[1] - 200], fill=(0, 0, 0), width=3)
    d.text((size[0] * 0.62, size[1] - 185), "Registrar", fill=(30, 20, 20), font=_font(26))
    return img


def image_bytes(img: Image.Image, fmt="PNG", quality=92) -> bytes:
    b = io.BytesIO()
    img.save(b, fmt, quality=quality) if fmt == "JPEG" else img.save(b, fmt)
    return b.getvalue()


def low_quality_scan(img: Image.Image) -> Image.Image:
    small = img.resize((img.width // 6, img.height // 6)).filter(ImageFilter.GaussianBlur(2))
    return small


def certificate_pdf(lines=CERT_LINES, pages=1, white_box=False, extra_text: str | None = None) -> bytes:
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.pdfgen import canvas

    b = io.BytesIO()
    c = canvas.Canvas(b, pagesize=landscape(A4))
    W, H = landscape(A4)
    c.setTitle("Certificate")
    c.setAuthor("Northbridge Institute of Technology")
    for p in range(pages):
        c.setStrokeColorRGB(0.34, 0.11, 0.14)
        c.setLineWidth(6)
        c.rect(20, 20, W - 40, H - 40)
        y = H - 90
        name_y = None
        for i, (text, fs, bold) in enumerate(lines):
            c.setFont("Times-Bold" if bold else "Times-Roman", fs * 0.45)
            c.drawCentredString(W / 2, y, text)
            if i == 3:
                name_y = y
            y -= fs * 0.95
        if pages > 1:
            c.setFont("Times-Roman", 10)
            c.drawCentredString(W / 2, 40, f"Page {p + 1} of {pages}")
        if white_box and p == 0:
            # Cover the recipient's name with a white rectangle and write a new one on top:
            # a classic manual-edit pattern.
            c.setFillColorRGB(1, 1, 1)
            c.rect(W / 2 - 120, name_y - 7, 240, 30, stroke=0, fill=1)
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 22)
            c.drawCentredString(W / 2, name_y, "Rahul Verma")
        if extra_text:
            c.setFont("Helvetica", 12)
            c.drawString(60, 60, extra_text)
        c.showPage()
    c.save()
    return b.getvalue()


# --------------------------------------------------------------------------- test PKI

@lru_cache(maxsize=None)
def test_pki(key_bits: int = 2048):
    now = dt.datetime.now(dt.timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=key_bits)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Hallmark Test Root CA"),
                         x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Hallmark Tests")])
    ca_cert = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
               .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
               .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=365))
               .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
               .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                            content_commitment=False, key_encipherment=False,
                                            data_encipherment=False, key_agreement=False,
                                            encipher_only=False, decipher_only=False), critical=True)
               .sign(ca_key, hashes.SHA256()))
    signer_key = rsa.generate_private_key(public_exponent=65537, key_size=key_bits)
    signer_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Northbridge Registrar (TEST)"),
                             x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Northbridge Institute (TEST)")])
    signer_cert = (x509.CertificateBuilder().subject_name(signer_name).issuer_name(ca_name)
                   .public_key(signer_key.public_key()).serial_number(x509.random_serial_number())
                   .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=180))
                   .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                   .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=True,
                                                key_cert_sign=False, crl_sign=False, key_encipherment=False,
                                                data_encipherment=False, key_agreement=False,
                                                encipher_only=False, decipher_only=False), critical=True)
                   .sign(ca_key, hashes.SHA256()))
    return ca_key, ca_cert, signer_key, signer_cert


def sign_pdf(pdf: bytes, md_algorithm: str = "sha256") -> bytes:
    from pyhanko.sign import signers
    from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter

    _, ca_cert, signer_key, signer_cert = test_pki()
    with tempfile.TemporaryDirectory() as td:
        kp, cp, cap = Path(td, "k.pem"), Path(td, "c.pem"), Path(td, "ca.pem")
        kp.write_bytes(signer_key.private_bytes(serialization.Encoding.PEM,
                                                serialization.PrivateFormat.PKCS8,
                                                serialization.NoEncryption()))
        cp.write_bytes(signer_cert.public_bytes(serialization.Encoding.PEM))
        cap.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
        signer = signers.SimpleSigner.load(str(kp), str(cp), ca_chain_files=(str(cap),))
    w = IncrementalPdfFileWriter(io.BytesIO(pdf))
    out = io.BytesIO()
    signers.sign_pdf(w, signers.PdfSignatureMetadata(field_name="IssuerSignature", md_algorithm=md_algorithm),
                     signer=signer, output=out)
    return out.getvalue()


def tamper_signed_bytes(signed: bytes) -> bytes:
    """Flip text inside the signed byte range (content stream unchanged in length)."""
    b = bytearray(signed)
    i = b.find(b"Priya")
    if i == -1:
        # content streams may be compressed; flip a byte early in the file instead
        i = 200
        b[i] = (b[i] + 1) % 256
    else:
        b[i:i + 5] = b"Pryia"
    return bytes(b)


def modify_after_signing(signed: bytes) -> bytes:
    """Append an incremental update that adds new visible content after signing."""
    from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
    from pyhanko.pdf_utils import generic
    from pyhanko.pdf_utils.content import RawContent

    w = IncrementalPdfFileWriter(io.BytesIO(signed))
    page_ref = w.root["/Pages"]["/Kids"][0]
    stream = generic.StreamObject(stream_data=b"BT /F1 24 Tf 100 100 Td (Grade: A+) Tj ET")
    ref = w.add_object(stream)
    page = page_ref.get_object()
    contents = page["/Contents"]
    arr = generic.ArrayObject(list(contents) if isinstance(contents, generic.ArrayObject) else [page.raw_get("/Contents")])
    arr.append(ref)
    page["/Contents"] = arr
    w.update_container(page)
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


def anchor_dir_with_test_ca() -> Path:
    _, ca_cert, _, _ = test_pki()
    d = Path(tempfile.mkdtemp(prefix="hallmark-anchors-"))
    (d / "test_ca.pem").write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    return d
