"""Field extraction and consistency checks for certificate text.

Design rules (from the Hallmark regression history):
* Certificate identifiers are extracted only when they sit next to an explicit
  label ("Certificate ID", "Credential No.", "Serial Number", ...). Words such as
  ``CERTIFICATE`` can therefore never be split into fake IDs like ``IFICATE``.
* Every candidate keeps provenance: source text, label, and why it was accepted
  or rejected.
* Values are only compared when they belong to the same semantic field.
* IDs are displayed but never used in the authenticity score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from dateutil import parser as dateparser

# --------------------------------------------------------------------------- IDs

ID_LABEL = re.compile(
    r"""(?ix)
    (?<![A-Za-z])
    (?P<label>
        (?:certificate|certification|credential|cert\.?|award|registration|reg\.?|
           enrol(?:l)?ment|roll|serial|verification|document|license|licence)
        \s*
        (?:id|identifier|code|no\.?|number|num\.?|\#)
      |
        (?:serial|reg(?:istration)?|roll)\s*(?:no\.?|number|\#)
      |
        (?:id|code|ref(?:erence)?)\s*(?:no\.?|number|\#)?
    )
    (?![A-Za-z])
    \s*[:\-–=#]?\s*
    (?P<value>[A-Za-z0-9][A-Za-z0-9\-/._]{2,40})
    """
)

# Common words that must never be accepted as identifiers even after a label.
STOP_WORDS = {
    "certificate", "certification", "certify", "certifies", "certified", "credential",
    "number", "code", "this", "that", "the", "and", "for", "has", "have", "is", "of",
    "awarded", "issued", "date", "name", "verify", "verification", "online", "at",
    "completion", "achievement", "participation", "here", "below", "above", "none", "n/a",
}


@dataclass
class IdCandidate:
    value: str
    label: str
    source: str
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _normalise_label(label: str) -> str:
    l = re.sub(r"[\s.#]+", " ", label.lower()).strip()
    return l


def _validate_id(value: str) -> tuple[bool, str]:
    v = value.strip(".-/_")
    if len(v) < 4:
        return False, "too short to be a reliable identifier"
    if v.lower() in STOP_WORDS:
        return False, "ordinary word, not an identifier"
    if not re.search(r"\d", v):
        # Purely alphabetic tokens are only accepted when they look like codes
        # (mostly upper-case, no vowel-heavy dictionary-word shape).
        if not (v.isupper() and len(v) >= 6 and sum(c in "AEIOU" for c in v) <= len(v) // 4):
            return False, "alphabetic token without digits; likely a word or heading fragment"
    return True, "labelled identifier with valid shape"


def extract_ids(text: str) -> list[IdCandidate]:
    out: list[IdCandidate] = []
    for line in text.splitlines():
        for m in ID_LABEL.finditer(line):
            value = m.group("value").rstrip(".,;:")
            ok, reason = _validate_id(value)
            out.append(IdCandidate(value=value, label=_normalise_label(m.group("label")),
                                   source=line.strip()[:160], accepted=ok, reason=reason))
    return out


# --------------------------------------------------------------------------- dates

MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_PATTERNS = [
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTHS}\.?,?\s+\d{{4}}\b",
    rf"\b{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
    rf"\b{MONTHS}\.?,?\s+\d{{4}}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}\b",
]
DATE_RE = re.compile("|".join(f"(?:{p})" for p in DATE_PATTERNS), re.I)

DATE_ROLES = [
    ("expiry", re.compile(r"(?i)\b(valid\s+(?:until|till|through|thru)|expir(?:y|es|ation)|valid\s+upto)\b")),
    ("issue", re.compile(r"(?i)\b(issued?(?:\s+on)?|date\s+of\s+issue|awarded\s+on|dated?)\b")),
    ("completion", re.compile(r"(?i)\b(completed?|completion|conferred|held\s+on|on)\b")),
    ("start", re.compile(r"(?i)\b(from|start(?:ed|ing)?|commenc\w+)\b")),
]


@dataclass
class DateMention:
    raw: str
    value: date | None
    role: str
    source: str
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"raw": self.raw, "value": self.value.isoformat() if self.value else None,
                "role": self.role, "source": self.source, "ambiguous": self.ambiguous}


def _parse_date(raw: str) -> tuple[date | None, bool]:
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", raw, flags=re.I).replace(" of ", " ")
    ambiguous = False
    m = re.fullmatch(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", s.strip())
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        ambiguous = a <= 12 and b <= 12 and a != b
    try:
        d = dateparser.parse(s, dayfirst=True, default=datetime(1900, 1, 1)).date()
    except (ValueError, OverflowError):
        return None, ambiguous
    if d.year < 1900 or d.year > 2200:
        return None, ambiguous
    return d, ambiguous


def extract_dates(text: str) -> list[DateMention]:
    out: list[DateMention] = []
    for line in text.splitlines():
        for m in DATE_RE.finditer(line):
            raw = m.group(0)
            prefix = line[: m.start()][-40:]
            role = "unspecified"
            for name, rx in DATE_ROLES:
                if rx.search(prefix):
                    role = name
                    break
            value, amb = _parse_date(raw)
            out.append(DateMention(raw=raw, value=value, role=role, source=line.strip()[:160], ambiguous=amb))
    return out


# --------------------------------------------------------------------------- names / issuer / title

ORG_WORDS = r"(?:university|institute|institution|college|academy|school|board|council|society|foundation|" \
            r"association|corporation|company|ltd|limited|inc|llc|pvt|private|technologies|solutions|" \
            r"centre|center|department|ministry|federation|organisation|organization|chapter)"
ORG_RE = re.compile(rf"(?i)^(?!.*\bcertif)[A-Z][\w&.,'()\- ]{{2,80}}\b{ORG_WORDS}\b[\w&.,'()\- ]{{0,60}}$")

RECIPIENT_CUES = re.compile(
    r"(?i)(?:this\s+is\s+to\s+certify\s+that|certify\s+that|certifies\s+that|presented\s+to|awarded\s+to|"
    r"granted\s+to|is\s+hereby\s+awarded\s+to|conferred\s+(?:up)?on|proudly\s+presented\s+to|recognizes|"
    r"recognises|in\s+recognition\s+of)\s*[:\-]?\s*(?P<rest>.*)$")
NAME_RE = re.compile(r"^(?:(?:Mr|Ms|Mrs|Dr|Shri|Smt|Kumari)\.?\s+)?([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-.]+){0,4})")
TITLE_RE = re.compile(r"(?i)\bcertificate\s+of\s+[a-z ]{3,40}|\b(?:bachelor|master|doctor|diploma)\s+of\s+[a-z ,&]{3,60}")
COURSE_CUES = re.compile(
    r"(?i)(?:successfully\s+completed?(?:\s+the)?|for\s+completing(?:\s+the)?|completion\s+of(?:\s+the)?|"
    r"has\s+completed(?:\s+the)?|for\s+(?:the\s+)?course|in\s+the\s+course|participat\w+\s+in(?:\s+the)?|"
    r"for\s+(?:his|her|their)\s+participation\s+in|degree\s+of|for\s+achieving)\s*[:\-]?\s*(?P<rest>.*)$")

PLACEHOLDERS = re.compile(
    r"(?i)\b(lorem\s+ipsum|your\s+name(\s+here)?|recipient\s+name|\[name\]|<name>|name\s+surname|"
    r"john\s+doe|jane\s+doe|course\s+name|xxxx+|sample\s+certificate|specimen)\b")


def _first_meaningful(lines: list[str], start: int) -> str | None:
    for j in range(start, min(start + 3, len(lines))):
        t = lines[j].strip(" .:-")
        if len(t) >= 3:
            return t
    return None


@dataclass
class ParsedCertificate:
    recipient: str | None = None
    issuer_candidates: list[str] = field(default_factory=list)
    title: str | None = None
    course: str | None = None
    dates: list[DateMention] = field(default_factory=list)
    id_candidates: list[IdCandidate] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)

    @property
    def accepted_ids(self) -> list[IdCandidate]:
        return [c for c in self.id_candidates if c.accepted]

    @property
    def issuer(self) -> str | None:
        return self.issuer_candidates[0] if self.issuer_candidates else None

    def fields_found(self) -> int:
        return sum(bool(x) for x in (self.recipient, self.issuer, self.title or self.course,
                                     any(d.value for d in self.dates)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipient": self.recipient,
            "issuer": self.issuer,
            "issuer_candidates": self.issuer_candidates[:5],
            "title": self.title,
            "course": self.course,
            "dates": [d.to_dict() for d in self.dates],
            "identifiers": [c.to_dict() for c in self.accepted_ids],
            "rejected_identifier_candidates": [c.to_dict() for c in self.id_candidates if not c.accepted],
            "placeholders": self.placeholders,
            "domains": self.domains,
        }


def parse_certificate(text: str) -> ParsedCertificate:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    pc = ParsedCertificate()

    for i, line in enumerate(lines):
        if pc.recipient is None:
            m = RECIPIENT_CUES.search(line)
            if m:
                rest = m.group("rest").strip(" .:-") or _first_meaningful(lines, i + 1) or ""
                nm = NAME_RE.match(rest)
                if nm:
                    pc.recipient = nm.group(1).strip()
        if pc.course is None:
            m = COURSE_CUES.search(line)
            if m:
                rest = m.group("rest").strip(" .:-\"'")
                if rest.lower() in ("", "course", "the course", "program", "programme", "the program",
                                    "the programme", "training", "the training"):
                    rest = _first_meaningful(lines, i + 1) or ""
                rest = re.split(r"(?i)\s+(?:on|dated|with|from)\s+\d", rest)[0].strip(" .\"'")
                if 3 <= len(rest) <= 120:
                    pc.course = rest
        if pc.title is None:
            m = TITLE_RE.search(line)
            if m:
                pc.title = re.sub(r"\s+", " ", m.group(0)).strip().title()
        if ORG_RE.match(line):
            org = re.sub(r"(?i)^(?:issued|awarded|presented|offered|certified)\s+by\s*[:\-]?\s*", "",
                         re.sub(r"\s+", " ", line)).strip()
            if org and org not in pc.issuer_candidates:
                pc.issuer_candidates.append(org)
        for ph in PLACEHOLDERS.finditer(line):
            pc.placeholders.append(ph.group(0))

    if pc.title is None:
        for line in lines[:8]:
            if re.search(r"(?i)\bcertificate\b", line) and len(line) <= 60:
                pc.title = re.sub(r"\s+", " ", line).strip().title()
                break

    pc.dates = extract_dates(text)
    pc.id_candidates = extract_ids(text)
    pc.domains = sorted(set(re.sub(r"^www\.", "", m.group(0).lower()) for m in re.finditer(
        r"\b(?:[a-z0-9-]+\.)+(?:com|org|edu|ac\.in|edu\.in|in|net|gov|gov\.in|io|ac\.uk|co\.uk)\b", text, re.I)))
    return pc


# --------------------------------------------------------------------------- consistency

@dataclass
class ConsistencyIssue:
    code: str
    message: str
    severity: str        # "info" | "warning" | "contradiction"
    data: dict[str, Any] = field(default_factory=dict)


def check_consistency(pc: ParsedCertificate, today: date | None = None) -> list[ConsistencyIssue]:
    today = today or date.today()
    issues: list[ConsistencyIssue] = []

    # 1) Identifiers: compare only values sharing the same normalised label.
    by_label: dict[str, set[str]] = {}
    for c in pc.accepted_ids:
        by_label.setdefault(c.label, set()).add(c.value.upper())
    for label, values in by_label.items():
        if len(values) > 1:
            issues.append(ConsistencyIssue(
                "conflicting_identifier",
                f"The field '{label}' appears with different values: {', '.join(sorted(values))}.",
                "contradiction", {"label": label, "values": sorted(values)}))

    # 2) Dates.
    dated = [d for d in pc.dates if d.value]
    future = [d for d in dated if d.role in ("issue", "completion") and d.value > today]
    for d in future:
        issues.append(ConsistencyIssue("future_issue_date",
                                       f"The {d.role} date {d.raw} is in the future.", "contradiction",
                                       {"date": d.value.isoformat()}))
    issue_dates = {d.value for d in dated if d.role == "issue"}
    if len(issue_dates) > 1:
        issues.append(ConsistencyIssue("multiple_issue_dates",
                                       "Different issue dates are stated: " +
                                       ", ".join(sorted(x.isoformat() for x in issue_dates)) + ".",
                                       "contradiction"))
    issue = min(issue_dates) if issue_dates else None
    for d in dated:
        if d.role == "expiry" and issue and d.value < issue:
            issues.append(ConsistencyIssue("expiry_before_issue",
                                           f"The expiry date {d.raw} is earlier than the issue date.",
                                           "contradiction"))
        if d.role == "start" and issue and d.value > issue:
            issues.append(ConsistencyIssue("start_after_issue",
                                           f"A start date {d.raw} falls after the issue date.", "warning"))
        if d.role == "expiry" and d.value < today:
            issues.append(ConsistencyIssue("expired",
                                           f"The certificate states it was valid until {d.raw}, which has passed.",
                                           "info"))
    if any(d.ambiguous for d in dated):
        issues.append(ConsistencyIssue("ambiguous_date_format",
                                       "A numeric date could be read as day/month or month/day; "
                                       "it was interpreted as day-first.", "info"))

    # 3) Placeholder / template text.
    if pc.placeholders:
        issues.append(ConsistencyIssue("placeholder_text",
                                       "Template placeholder text was found: " +
                                       ", ".join(sorted(set(p.lower() for p in pc.placeholders))) + ".",
                                       "contradiction"))
    return issues
