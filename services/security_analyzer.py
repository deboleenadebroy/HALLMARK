"""File-level facts: SHA-256 fingerprint, PDF revision history and active content.

These facts feed the integrity analyser and the report. A SHA-256 hash only
identifies the exact bytes analysed; without a trusted reference hash it says
nothing about authenticity.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

ACTIVE_CONTENT = {
    "/JavaScript": "embedded JavaScript",
    "/JS": "embedded JavaScript",
    "/Launch": "a launch action",
    "/EmbeddedFile": "an embedded file",
    "/OpenAction": "an automatic open action",
    "/RichMedia": "rich media",
    "/SubmitForm": "a form-submission action",
}


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pdf_revision_count(data: bytes) -> int:
    """Number of saved revisions (each incremental update ends with %%EOF)."""
    return max(1, len(re.findall(rb"%%EOF", data)))


def pdf_active_content(data: bytes) -> list[str]:
    found = []
    for key, label in ACTIVE_CONTENT.items():
        if re.search(re.escape(key.encode()) + rb"(?![A-Za-z])", data):
            if label not in found:
                found.append(label)
    return found


def file_facts(doc) -> dict[str, Any]:
    facts: dict[str, Any] = {"sha256": sha256_hex(doc.data), "size_bytes": len(doc.data), "kind": doc.kind}
    if doc.is_pdf:
        facts["revisions"] = pdf_revision_count(doc.data)
        facts["active_content"] = pdf_active_content(doc.data)
    return facts
