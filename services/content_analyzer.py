"""Evidence builders for extracted content: identity & details, issuer, registry."""
from __future__ import annotations

import re
from typing import Any

from .certificate_parser import ParsedCertificate, check_consistency, parse_certificate
from .evidence import Evidence, Finding, Status, Strength

CERT_KEYWORDS = re.compile(
    r"(?i)\b(certif\w*|award\w*|diploma|degree|completion|achievement|participation|recognition|"
    r"presented|conferred|accomplishment|appreciation|excellence|honou?r|merit|transcript|graduat\w*|"
    r"qualification|licen[cs]e|credential|course|training|workshop|internship|"
    r"grade\s*card|marks?\s*card|mark\s*sheet|marksheet|semester|examination|cgpa|sgpa|university|"
    r"institute|board)\b")

SEVERITY = {"contradiction": (-1, Strength.MODERATE), "warning": (-1, Strength.WEAK), "info": (0, Strength.NONE)}


def looks_like_certificate(text: str) -> bool:
    return len(set(m.group(0).lower() for m in CERT_KEYWORDS.finditer(text))) >= 2


def analyse_identity(page_texts: dict[int, str], text_quality: str) -> tuple[Evidence, ParsedCertificate]:
    full = "\n".join(page_texts[k] for k in sorted(page_texts))
    pc = parse_certificate(full)
    limitations = ["Extracted text may contain OCR errors in names, dates and codes.",
                   "Consistent details show internal coherence only; they do not prove the certificate was issued.",
                   "Certificate identifiers are displayed for reference and never used in the score."]
    details: dict[str, Any] = {"extracted": pc.to_dict(), "text_quality": text_quality}

    if text_quality in ("unreadable", "unavailable") or len(full.split()) < 5:
        return Evidence("identity_and_details", Status.INCONCLUSIVE,
                        "Too little text could be read to check the certificate's details.",
                        limitations=limitations, details=details), pc

    findings: list[Finding] = []
    for issue in check_consistency(pc):
        direction, strength = SEVERITY[issue.severity]
        if issue.code == "placeholder_text":
            strength = Strength.MODERATE
        findings.append(Finding(issue.code, issue.message, direction, strength, data=issue.data))

    # Cross-page recipient consistency (only when each page names a recipient).
    names = {}
    for n, t in page_texts.items():
        r = parse_certificate(t).recipient
        if r:
            names[n] = r
    if len(set(v.lower() for v in names.values())) > 1:
        findings.append(Finding("recipient_differs_across_pages",
                                "Different recipient names appear on different pages: " +
                                "; ".join(f"page {k}: {v}" for k, v in sorted(names.items())) + ".",
                                -1, Strength.WEAK, data={"names": names}))

    negatives = [f for f in findings if f.direction < 0]
    found = pc.fields_found()
    if any(f.strength == Strength.MODERATE for f in negatives):
        status, expl = Status.FAIL, "The certificate's details contain internal contradictions."
    elif negatives:
        status, expl = Status.INCONCLUSIVE, "Minor inconsistencies were found in the certificate's details."
    elif found >= 3:
        status = Status.PASS
        expl = f"{found} of 4 key fields (recipient, issuer, title/course, date) were found and are mutually consistent."
        findings.append(Finding("details_consistent", expl, +1, Strength.WEAK))
    else:
        status = Status.INCONCLUSIVE
        expl = f"Only {found} of 4 key fields could be identified, so consistency could not be assessed well."
    if text_quality == "poor":
        limitations.append("OCR confidence was low, so findings here are less reliable.")
    return Evidence("identity_and_details", status, expl, findings=findings,
                    limitations=limitations, details=details), pc


def analyse_issuer(pc: ParsedCertificate) -> Evidence:
    details = {"apparent_issuer": pc.issuer, "issuer_candidates": pc.issuer_candidates[:5], "domains": pc.domains}
    who = f"Apparent issuer from the document text: {pc.issuer}." if pc.issuer else \
        "No issuer name could be identified in the document text."
    return Evidence("issuer_discovery", Status.NOT_PERFORMED,
                    "Issuer verification against external sources is not performed — Hallmark analyses offline. " + who,
                    limitations=["An issuer name printed on a certificate is a claim, not verification."],
                    details=details)


def analyse_registry() -> Evidence:
    return Evidence("registry", Status.NOT_PERFORMED,
                    "No institutional registry lookup was performed. Hallmark does not contact issuers or "
                    "external databases, so the certificate could not be matched against official records.",
                    limitations=["Only the issuing institution can confirm that a certificate was issued."])
