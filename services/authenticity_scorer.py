"""Estimated Authenticity Score.

How the score is built
----------------------
Evidence is combined additively in log-odds space, starting from a neutral
prior of 0 (50%). Each finding contributes a bounded amount set by its
strength and category (``WEIGHTS``). The result is mapped back with the
logistic function and shown as a 0–100% *Estimated Authenticity Score*.

This is a documented heuristic, NOT a calibrated probability. Only the
visual-model component is calibrated (temperature scaling on validation data);
the other weights are product defaults chosen so that:

* heuristic checks alone can never push the score above ~60%, and *weak*
  heuristic indicators alone (however many) cannot push it below ~29%
  (``WEAK_NEGATIVE_CAP``). Only moderate heuristic findings — e.g. covered
  text or template placeholder text — can reach the lower bands;
* a valid, trusted signature is the only way to reach the "strong evidence"
  band; a broken signature is the strongest negative signal;
* missing evidence (no signature, no model, unreadable text) contributes 0 —
  it is never treated as negative evidence.

Double counting is limited by capping each category, and by halving the
pixel-noise finding when the visual model (which also sees the pixels) ran.

If strong positive and strong negative findings coexist, the conflict is
flagged and the total is shrunk toward 50% instead of letting one side win.

The score is withheld (``None``) when there is too little evidence for any
number to be meaningful: unreadable documents, or documents that do not look
like certificates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .evidence import Evidence, Status, Strength

# log-odds contribution per finding, by category and strength (signed by direction)
WEIGHTS = {
    "cryptography": {Strength.STRONG: 2.5, Strength.MODERATE: 0.8, Strength.WEAK: 0.3},
    "document_integrity": {Strength.STRONG: 2.0, Strength.MODERATE: 0.9, Strength.WEAK: 0.35},
    "identity_and_details": {Strength.STRONG: 1.6, Strength.MODERATE: 0.8, Strength.WEAK: 0.3},
}
NEGATIVE_OVERRIDES = {"signed_content_altered": -3.0, "signature_invalid": -3.0,
                      "modified_after_signing": -2.5, "signature_error": -2.0}
POSITIVE_SMALL = {"no_integrity_issues": 0.2, "details_consistent": 0.2,
                  # Anyone can self-sign a PDF: an untrusted signature proves the bytes are unchanged
                  # since signing, not who signed it, so it earns only a small boost.
                  "signature_valid_untrusted": 0.4}
CATEGORY_CAPS = {   # (min, max) total contribution per category
    "cryptography": (-3.0, 2.5),
    "document_integrity": (-2.0, 0.2),
    "identity_and_details": (-1.6, 0.2),
    "visual_model": (-3.0, 3.0),
}
VISUAL_MARGIN_CLIP = 3.0
CONFLICT_SHRINK = 0.5
WEAK_NEGATIVE_CAP = -0.9   # total from all weak negative heuristic findings

BANDS = [  # (lower bound, key, label, description)
    (90, "strong_authentic", "Strong evidence of authenticity", "Strong evidence consistent with authenticity."),
    (70, "likely_authentic", "Likely authentic", "Evidence generally favours authenticity, but uncertainty remains."),
    (40, "inconclusive", "Inconclusive", "Inconclusive; additional review is recommended."),
    (20, "suspicious", "Suspicious", "Significant suspicious evidence."),
    (0, "likely_forged", "Likely forged", "Strong evidence consistent with forgery."),
]


@dataclass
class Contribution:
    category: str
    code: str
    message: str
    logit: float

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "code": self.code, "message": self.message,
                "logit": round(self.logit, 3)}


@dataclass
class ScoreResult:
    score: float | None
    band: dict[str, str]
    total_logit: float
    contributions: list[Contribution] = field(default_factory=list)
    conflict: bool = False
    withheld_reason: str | None = None
    reliability: str = "low"
    method: str = ("Heuristic evidence combination (log-odds). Not a calibrated probability; "
                   "only the visual-model component is calibrated.")

    def to_dict(self) -> dict[str, Any]:
        return {"score": None if self.score is None else round(self.score, 1),
                "band": self.band, "total_logit": round(self.total_logit, 3),
                "contributions": [c.to_dict() for c in self.contributions],
                "conflict": self.conflict, "withheld_reason": self.withheld_reason,
                "reliability": self.reliability, "method": self.method}


def band_for(score: float | None) -> dict[str, str]:
    if score is None:
        return {"key": "unavailable", "label": "Inconclusive — score unavailable",
                "description": "There is not enough reliable evidence to estimate authenticity."}
    for lo, key, label, desc in BANDS:
        if score >= lo:
            return {"key": key, "label": label, "description": desc}
    return {"key": BANDS[-1][1], "label": BANDS[-1][2], "description": BANDS[-1][3]}


def _finding_logit(category: str, f, visual_ran: bool) -> float:
    if f.direction == 0 or f.strength == Strength.NONE:
        return 0.0
    if f.code in NEGATIVE_OVERRIDES:
        return NEGATIVE_OVERRIDES[f.code]
    if f.code in POSITIVE_SMALL:
        return POSITIVE_SMALL[f.code]
    w = WEIGHTS.get(category, {}).get(f.strength, 0.0) * f.direction
    if f.code == "noise_inconsistency" and visual_ran:
        w *= 0.5      # the visual model also sees pixel content; avoid counting it twice
    return w


def compute_score(evidence: list[Evidence], *, in_scope: bool = True, readable: bool = True) -> ScoreResult:
    by_cat = {e.category: e for e in evidence}
    visual = by_cat.get("visual_model")
    visual_ran = bool(visual and visual.status in (Status.PASS, Status.FAIL))
    contributions: list[Contribution] = []
    cat_totals: dict[str, float] = {}
    weak_neg_total = 0.0

    for e in evidence:
        if not e.performed:
            continue
        if e.category == "visual_model":
            if e.status not in (Status.PASS, Status.FAIL):
                continue   # inconclusive / out-of-distribution: no contribution
            pf = float(e.details.get("document_p_forged", 0.5))
            pf = min(max(pf, 1e-6), 1 - 1e-6)
            margin = math.log(pf / (1 - pf))
            scale = 0.5 if e.details.get("document_ood") == "borderline" else 1.0
            lg = -max(-VISUAL_MARGIN_CLIP, min(VISUAL_MARGIN_CLIP, margin)) * scale
            contributions.append(Contribution("visual_model", "visual_prediction", e.explanation, lg))
            cat_totals["visual_model"] = lg
            continue
        if e.category not in CATEGORY_CAPS:
            continue
        tot = 0.0
        for f in e.findings:
            lg = _finding_logit(e.category, f, visual_ran)
            if lg < 0 and f.strength == Strength.WEAK and f.code not in NEGATIVE_OVERRIDES:
                allowed = max(lg, WEAK_NEGATIVE_CAP - weak_neg_total)
                weak_neg_total += allowed
                lg = allowed
            if lg:
                contributions.append(Contribution(e.category, f.code, f.message, lg))
                tot += lg
        lo, hi = CATEGORY_CAPS[e.category]
        cat_totals[e.category] = max(lo, min(hi, tot))

    total = sum(cat_totals.values())
    strong_pos = any(c.logit >= 0.8 for c in contributions)
    strong_neg = any(c.logit <= -0.8 for c in contributions)
    conflict = strong_pos and strong_neg
    if conflict:
        total *= CONFLICT_SHRINK

    crypto = by_cat.get("cryptography")
    decisive = visual_ran or (crypto is not None and crypto.status in (Status.PASS, Status.FAIL))

    withheld = None
    if not readable:
        withheld = "The document could not be read reliably, so no score is given."
    elif not in_scope and not decisive:
        withheld = ("The document does not appear to be a certificate, which is outside Hallmark's "
                    "validated scope, so no score is given.")
    if withheld:
        return ScoreResult(None, band_for(None), total, contributions, conflict, withheld, "unavailable")

    score = 100.0 / (1.0 + math.exp(-total))
    if decisive and not conflict:
        reliability = "moderate" if visual_ran and not (crypto and crypto.status == Status.PASS) else "higher"
    else:
        reliability = "low"
    contributions.sort(key=lambda c: -abs(c.logit))
    return ScoreResult(score, band_for(score), total, contributions, conflict, None, reliability)
