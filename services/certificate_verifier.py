"""Cryptographic verification.

* Embedded PDF signatures are inspected automatically (pyHanko): integrity of
  the signed byte range, cryptographic validity, signature algorithm and
  padding, X.509 signer certificate details, trust against locally configured
  anchors (``trust_anchors/``) and modifications made after signing.
* ``verify_rsa_signature`` is a standalone, strict RSA verifier (PKCS#1 v1.5 /
  PSS with SHA-256/384/512 only) used by the PDF path and by tests.
* Everything runs offline: no CRL/OCSP fetching. Revocation is therefore
  reported as not checked.

A mathematically valid signature only proves the signed bytes match the
signer's key. It does not by itself prove that a legitimate institution issued
the certificate; that requires trust in the signer's certificate.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

from .evidence import Evidence, Finding, Status, Strength

log = logging.getLogger(__name__)

TRUST_ANCHOR_DIR = Path(__file__).resolve().parent.parent / "trust_anchors"
MIN_RSA_BITS = 2048
ALLOWED_HASHES = {"sha256": hashes.SHA256, "sha384": hashes.SHA384, "sha512": hashes.SHA512}
WEAK_HASHES = {"md5", "sha1", "sha224"}


class UnsupportedAlgorithm(ValueError):
    pass


# --------------------------------------------------------------------------- RSA primitive

def verify_rsa_signature(public_key: RSAPublicKey | bytes, data: bytes, signature: bytes,
                         hash_name: str = "sha256", padding: str = "pkcs1v15") -> bool:
    """Strict RSA signature verification.

    Returns True/False for valid/invalid signatures. Raises ``UnsupportedAlgorithm``
    for weak hashes, unknown padding, non-RSA keys, or keys shorter than 2048 bits.
    """
    if isinstance(public_key, (bytes, bytearray)):
        try:
            public_key = serialization.load_pem_public_key(bytes(public_key))
        except ValueError:
            public_key = serialization.load_der_public_key(bytes(public_key))
    if not isinstance(public_key, RSAPublicKey):
        raise UnsupportedAlgorithm("Key is not an RSA public key.")
    if public_key.key_size < MIN_RSA_BITS:
        raise UnsupportedAlgorithm(f"RSA key of {public_key.key_size} bits is below the {MIN_RSA_BITS}-bit minimum.")
    h = hash_name.lower().replace("-", "")
    if h not in ALLOWED_HASHES:
        raise UnsupportedAlgorithm(f"Hash algorithm '{hash_name}' is not accepted.")
    algo = ALLOWED_HASHES[h]()
    if padding == "pkcs1v15":
        pad = rsa_padding.PKCS1v15()
    elif padding == "pss":
        pad = rsa_padding.PSS(mgf=rsa_padding.MGF1(algo), salt_length=rsa_padding.PSS.AUTO)
    else:
        raise UnsupportedAlgorithm(f"Padding '{padding}' is not supported.")
    try:
        public_key.verify(signature, data, pad, algo)
        return True
    except InvalidSignature:
        return False


# --------------------------------------------------------------------------- trust anchors

def load_trust_anchors(directory: Path | None = None) -> list[x509.Certificate]:
    directory = directory or TRUST_ANCHOR_DIR
    out: list[x509.Certificate] = []
    if not directory.exists():
        return out
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in (".pem", ".crt", ".cer", ".der"):
            continue
        raw = p.read_bytes()
        try:
            if b"-----BEGIN CERTIFICATE-----" in raw:
                out.extend(x509.load_pem_x509_certificates(raw))
            else:
                out.append(x509.load_der_x509_certificate(raw))
        except ValueError:
            log.warning("Ignoring unreadable trust anchor %s", p.name)
    return out


# --------------------------------------------------------------------------- PDF signatures

@dataclass
class SignatureReport:
    field_name: str
    intact: bool | None = None           # signed byte range digest matches
    valid: bool | None = None            # cryptographic signature verifies
    trusted: bool | None = None          # chains to a configured trust anchor
    algorithm: str | None = None
    digest_algorithm: str | None = None
    key_type: str | None = None
    key_bits: int | None = None
    signer_subject: str | None = None
    signer_issuer: str | None = None
    not_before: str | None = None
    not_after: str | None = None
    signing_time: str | None = None
    coverage: str | None = None
    modification_level: str | None = None
    unsupported: str | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _name(n: Any) -> str | None:
    try:
        return n.human_friendly
    except Exception:
        return str(n) if n is not None else None


def inspect_pdf_signatures(data: bytes, anchors: list[x509.Certificate] | None = None) -> list[SignatureReport]:
    from pyhanko.pdf_utils.reader import PdfFileReader
    from pyhanko.sign.validation import validate_pdf_signature
    from pyhanko_certvalidator import ValidationContext
    from asn1crypto import x509 as asn1_x509

    anchors = load_trust_anchors() if anchors is None else anchors
    roots = [asn1_x509.Certificate.load(c.public_bytes(serialization.Encoding.DER)) for c in anchors]

    reader = PdfFileReader(io.BytesIO(data), strict=False)
    reports: list[SignatureReport] = []
    for sig in reader.embedded_signatures:
        rep = SignatureReport(field_name=sig.field_name or "signature")
        try:
            vc = ValidationContext(trust_roots=roots, allow_fetching=False, revocation_mode="soft-fail")
            st = validate_pdf_signature(sig, vc)
            rep.intact = bool(st.intact)
            rep.valid = bool(st.valid)
            rep.trusted = bool(st.trusted) if roots else False
            rep.digest_algorithm = getattr(st, "md_algorithm", None)
            mech = getattr(st, "pkcs7_signature_mechanism", None)
            rep.algorithm = str(mech) if mech else None
            cert = st.signing_cert
            if cert is not None:
                rep.signer_subject = _name(cert.subject)
                rep.signer_issuer = _name(cert.issuer)
                rep.not_before = cert.not_valid_before.isoformat()
                rep.not_after = cert.not_valid_after.isoformat()
                pk = cert.public_key
                rep.key_type = pk.algorithm
                try:
                    rep.key_bits = int(pk.bit_size)
                except Exception:
                    pass
            if getattr(st, "signer_reported_dt", None):
                rep.signing_time = st.signer_reported_dt.isoformat()
            cov = getattr(st, "coverage", None)
            rep.coverage = cov.name if cov is not None else None
            ml = getattr(st, "modification_level", None)
            rep.modification_level = ml.name if ml is not None else None
            if not roots:
                rep.notes.append("No trust anchors are configured, so the signer could not be trusted.")
        except Exception as exc:  # pyHanko raises for unsupported/malformed structures
            msg = f"{exc.__class__.__name__}: {exc}"[:300]
            if "not supported" in msg.lower() or "unsupported" in msg.lower() or "NotImplemented" in msg:
                rep.unsupported = msg
            else:
                rep.error = msg
        # algorithm policy
        da = (rep.digest_algorithm or "").lower()
        if da in WEAK_HASHES:
            rep.notes.append(f"Weak digest algorithm {da.upper()} was used.")
        if rep.key_type == "rsa" and rep.key_bits and rep.key_bits < MIN_RSA_BITS:
            rep.notes.append(f"RSA key size {rep.key_bits} bits is below the {MIN_RSA_BITS}-bit minimum.")
        reports.append(rep)
    return reports


def analyse_cryptography(doc) -> Evidence:
    """Produce the ``cryptography`` evidence for a loaded document."""
    if not doc.is_pdf:
        return Evidence("cryptography", Status.NOT_APPLICABLE,
                        "Image files cannot carry an embedded digital signature, so no signature check applies. "
                        "This is not evidence against authenticity.",
                        limitations=["A scanned or photographed certificate can only be assessed visually."])
    try:
        reports = inspect_pdf_signatures(doc.data)
    except Exception as exc:
        return Evidence("cryptography", Status.INCONCLUSIVE,
                        "The PDF signature structure could not be read.",
                        findings=[Finding("signature_parse_error", str(exc)[:200])],
                        limitations=["Malformed PDF structures can prevent signature inspection."])
    if not reports:
        return Evidence("cryptography", Status.NOT_APPLICABLE,
                        "The PDF has no embedded digital signature. Many genuine certificates are unsigned, "
                        "so this is not evidence against authenticity.",
                        limitations=["Without a signature, document integrity cannot be proven cryptographically."])

    findings: list[Finding] = []
    any_fail = any_unsupported = False
    any_trusted_ok = any_ok = False
    anchors_configured = bool(load_trust_anchors())
    for r in reports:
        if r.unsupported:
            any_unsupported = True
            findings.append(Finding("signature_unsupported",
                                    f"Signature '{r.field_name}' uses an unsupported format: {r.unsupported}",
                                    0, Strength.NONE, data=r.to_dict()))
            continue
        if r.error:
            any_fail = True
            findings.append(Finding("signature_error",
                                    f"Signature '{r.field_name}' could not be validated: {r.error}",
                                    -1, Strength.STRONG, data=r.to_dict()))
            continue
        weak = [n for n in r.notes if n.startswith(("Weak", "RSA key size"))]
        if not r.intact:
            any_fail = True
            findings.append(Finding("signed_content_altered",
                                    f"Signature '{r.field_name}': the signed content no longer matches its digest — "
                                    "the document was altered after signing.", -1, Strength.STRONG, data=r.to_dict()))
        elif not r.valid:
            any_fail = True
            findings.append(Finding("signature_invalid",
                                    f"Signature '{r.field_name}' is cryptographically invalid.",
                                    -1, Strength.STRONG, data=r.to_dict()))
        else:
            mod = r.modification_level or ""
            if mod not in ("NONE", "LTA_UPDATES", ""):
                any_fail = True
                findings.append(Finding("modified_after_signing",
                                        f"Signature '{r.field_name}' is valid, but the file was changed after signing "
                                        f"(pyHanko modification level: {mod}).", -1, Strength.STRONG, data=r.to_dict()))
            elif weak:
                findings.append(Finding("weak_signature_algorithm",
                                        f"Signature '{r.field_name}' verifies but uses weak parameters: " + " ".join(weak),
                                        0, Strength.NONE, data=r.to_dict()))
            elif r.trusted:
                any_trusted_ok = any_ok = True
                findings.append(Finding("signature_valid_trusted",
                                        f"Signature '{r.field_name}' is valid, covers the signed content, and chains to a "
                                        f"configured trust anchor (signer: {r.signer_subject}).",
                                        +1, Strength.STRONG, data=r.to_dict()))
            else:
                any_ok = True
                findings.append(Finding("signature_valid_untrusted",
                                        f"Signature '{r.field_name}' is mathematically valid and the signed content is intact, "
                                        f"but the signer (\"{r.signer_subject}\") is not linked to a trusted authority.",
                                        +1, Strength.MODERATE, data=r.to_dict()))

    limitations = ["Revocation (CRL/OCSP) is not checked because analysis runs offline."]
    if not anchors_configured:
        limitations.append("No trust anchors are configured, so a valid signature cannot be tied to a legitimate issuer.")
    limitations.append("A valid signature proves the bytes are unchanged since signing, not that the signer is a legitimate institution.")

    if any_fail:
        status, expl = Status.FAIL, "At least one embedded signature failed verification or the document was changed after signing."
    elif any_trusted_ok:
        status, expl = Status.PASS, "The embedded digital signature is valid and trusted."
    elif any_ok:
        status, expl = Status.PASS, "The embedded signature is valid, but the signer's identity could not be independently trusted."
    elif any_unsupported:
        status, expl = Status.INCONCLUSIVE, "The document carries a signature in a format Hallmark cannot verify."
    else:
        status, expl = Status.INCONCLUSIVE, "The signature verified only with weak parameters, so it is not relied on."
    return Evidence("cryptography", status, expl, findings=findings, limitations=limitations,
                    details={"signatures": [r.to_dict() for r in reports],
                             "trust_anchors_configured": anchors_configured})
