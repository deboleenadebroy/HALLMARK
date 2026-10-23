"""Shared evidence data model used by every Hallmark analyser.

Every analyser returns one or more ``Evidence`` objects. The assessment engine
(``tier6_classifier``) and the scorer (``authenticity_scorer``) only ever read
these objects, so the UI can never show a result that the analysers did not
actually produce.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "PASS"                    # check ran and supports authenticity
    FAIL = "FAIL"                    # check ran and found a meaningful problem
    INCONCLUSIVE = "INCONCLUSIVE"    # check ran but could not reach a reliable conclusion
    NOT_APPLICABLE = "NOT_APPLICABLE"  # check does not apply to this file
    NOT_PERFORMED = "NOT_PERFORMED"  # check applies in principle but was not run

    @property
    def performed(self) -> bool:
        return self in (Status.PASS, Status.FAIL, Status.INCONCLUSIVE)


class Strength(str, Enum):
    """How much a finding should be allowed to move the score."""
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


# Evidence categories. The first five are Hallmark's original dimensions;
# ``visual_model`` is reported separately so the model output is never
# confused with the overall score.
CATEGORIES = (
    "registry",
    "cryptography",
    "document_integrity",
    "issuer_discovery",
    "identity_and_details",
    "visual_model",
)

CATEGORY_LABELS = {
    "registry": "Institutional registry",
    "cryptography": "Cryptographic signature",
    "document_integrity": "Document integrity",
    "issuer_discovery": "Issuer identification",
    "identity_and_details": "Identity & certificate details",
    "visual_model": "AI visual forgery model",
}


@dataclass
class Finding:
    """One concrete observation inside an evidence category."""
    code: str                 # stable machine identifier, e.g. "pdf_white_overlay"
    message: str              # human-readable explanation
    direction: int = 0        # +1 supports authenticity, -1 suggests forgery, 0 neutral/info
    strength: Strength = Strength.NONE
    page: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["strength"] = self.strength.value
        return d


@dataclass
class Evidence:
    category: str
    status: Status
    explanation: str
    findings: list[Finding] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown evidence category {self.category!r}")

    @property
    def performed(self) -> bool:
        return self.status.performed

    @property
    def label(self) -> str:
        return CATEGORY_LABELS[self.category]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "label": self.label,
            "status": self.status.value,
            "performed": self.performed,
            "explanation": self.explanation,
            "findings": [f.to_dict() for f in self.findings],
            "limitations": list(self.limitations),
            "details": self.details,
        }
