"""Final assessment (tier 6): classification, explanation and key findings.

Hallmark's six classification states are preserved:
VERIFIED, LIKELY_AUTHENTIC, INCONCLUSIVE, SUSPICIOUS, LIKELY_FRAUDULENT, REVOKED.

* VERIFIED requires a valid signature chained to a configured trust anchor and
  no conflicting negative evidence — visual evidence alone can never verify.
* REVOKED requires revocation information; Hallmark runs offline, so it is
  never assigned at present (kept for API compatibility).
* Otherwise the state follows the score band; a withheld score is INCONCLUSIVE.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from .authenticity_scorer import ScoreResult
from .evidence import Evidence, Status


class Classification(str, Enum):
    VERIFIED = "VERIFIED"
    LIKELY_AUTHENTIC = "LIKELY_AUTHENTIC"
    INCONCLUSIVE = "INCONCLUSIVE"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_FRAUDULENT = "LIKELY_FRAUDULENT"
    REVOKED = "REVOKED"


BAND_TO_CLASS = {
    "strong_authentic": Classification.LIKELY_AUTHENTIC,
    "likely_authentic": Classification.LIKELY_AUTHENTIC,
    "inconclusive": Classification.INCONCLUSIVE,
    "unavailable": Classification.INCONCLUSIVE,
    "suspicious": Classification.SUSPICIOUS,
    "likely_forged": Classification.LIKELY_FRAUDULENT,
}


def classify(evidence: list[Evidence], score: ScoreResult) -> dict[str, Any]:
    by = {e.category: e for e in evidence}
    crypto = by.get("cryptography")
    trusted = bool(crypto and crypto.status == Status.PASS and
                   any(f.code == "signature_valid_trusted" for f in crypto.findings))
    neg_strong = any(c.logit <= -0.8 for c in score.contributions)
    cls = BAND_TO_CLASS[score.band["key"]]
    if trusted and not neg_strong and score.score is not None and score.score >= 90:
        cls = Classification.VERIFIED

    explanation = _explain(evidence, score, cls)
    key = _key_findings(evidence, score)
    recs = _recommendations(evidence, score, cls)
    warnings = []
    if score.conflict:
        warnings.append("Strong evidence points in opposite directions. The score was pulled toward 50% "
                        "and the result needs human review.")
    for e in evidence:
        if e.status == Status.INCONCLUSIVE:
            warnings.append(f"{e.label}: {e.explanation}")
    return {"classification": cls.value, "explanation": explanation, "key_findings": key,
            "recommendations": recs, "warnings": warnings}


def _explain(evidence, score: ScoreResult, cls: Classification) -> str:
    by = {e.category: e for e in evidence}
    if score.score is None:
        return score.withheld_reason or "There is not enough reliable evidence to estimate authenticity."
    parts = []
    c = by.get("cryptography")
    if c and c.status == Status.PASS:
        parts.append("the embedded digital signature is valid" +
                     (" and trusted" if cls == Classification.VERIFIED else ", though its signer is not independently trusted"))
    elif c and c.status == Status.FAIL:
        parts.append("the embedded digital signature failed verification")
    v = by.get("visual_model")
    if v and v.status in (Status.PASS, Status.FAIL):
        parts.append(v.explanation[0].lower() + v.explanation[1:].rstrip("."))
    elif v and v.status == Status.NOT_PERFORMED:
        parts.append("the AI visual model was not used" if v.details.get("validation") else "no AI visual model was available")
    i = by.get("document_integrity")
    if i and i.status == Status.FAIL:
        parts.append("integrity checks found signs consistent with alteration")
    elif i and i.status == Status.INCONCLUSIVE and any(f.direction < 0 for f in i.findings):
        parts.append("integrity checks raised minor concerns")
    elif i and i.status == Status.PASS:
        parts.append("no signs of alteration were detected")
    d = by.get("identity_and_details")
    if d and d.status == Status.FAIL:
        parts.append("the certificate's details contradict each other")
    lead = {"strong_authentic": "Strong evidence is consistent with authenticity",
            "likely_authentic": "The evidence generally favours authenticity",
            "inconclusive": "The evidence is inconclusive",
            "suspicious": "Significant suspicious evidence was found",
            "likely_forged": "Strong evidence is consistent with forgery"}[score.band["key"]]
    text = lead + (": " + "; ".join(parts) + "." if parts else ".")
    if score.band["key"] == "inconclusive" and not any(
            e.status in (Status.PASS, Status.FAIL) for e in evidence if e.category in ("cryptography", "visual_model")):
        text += " Without a digital signature or a trained visual model, Hallmark can only run heuristic checks."
    if score.conflict:
        text += " Some findings conflict, so confidence was reduced."
    return text


def _key_findings(evidence, score: ScoreResult, n: int = 3) -> list[dict[str, Any]]:
    out = []
    contribs = score.contributions
    if any(abs(c.logit) >= 2.0 for c in contribs):
        # A decisive finding exists: don't dilute the headline with weak supporting signals.
        contribs = [c for c in contribs if abs(c.logit) >= 0.3]
    for c in contribs[:n]:
        out.append({"direction": "supports" if c.logit > 0 else "concern", "category": c.category,
                    "message": c.message})
    if len(out) < n:
        for e in evidence:
            if e.category in ("cryptography", "visual_model") and e.status in (Status.NOT_APPLICABLE, Status.NOT_PERFORMED):
                out.append({"direction": "neutral", "category": e.category, "message": e.explanation})
            if len(out) >= n:
                break
    return out[:n]


def _recommendations(evidence, score: ScoreResult, cls: Classification) -> list[str]:
    recs = []
    if cls != Classification.VERIFIED:
        recs.append("Confirm the certificate directly with the issuing institution using contact details "
                    "from its official website, not from the certificate itself.")
    if score.score is not None and score.score < 40:
        recs.append("Do not rely on this certificate until it has been independently confirmed.")
    if score.conflict or (score.score is not None and 40 <= score.score < 70):
        recs.append("Have a person review the detailed findings below.")
    by = {e.category: e for e in evidence}
    if by.get("visual_model") and by["visual_model"].status == Status.NOT_PERFORMED:
        recs.append("Install a trained visual model to add AI forgery detection (see README).")
    q = by.get("identity_and_details")
    if q and q.status == Status.INCONCLUSIVE:
        recs.append("If possible, upload a higher-quality scan or the original digital file.")
    return recs
