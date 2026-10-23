"""Upload validation and document loading (PDF and raster images).

Validation is based on file signatures ("magic bytes"), not on the
filename extension alone. Uploaded content is never executed.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_FILE_BYTES = 20 * 1024 * 1024        # 20 MB
MAX_PDF_PAGES = 20
RENDER_DPI = 200
MAX_IMAGE_PIXELS = 60_000_000

ALLOWED_EXTENSIONS = {".pdf": "pdf", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class DocumentError(Exception):
    """Raised when an upload cannot be analysed. ``code`` is machine-readable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Page:
    number: int                      # 1-based
    image: Image.Image               # RGB render / original image
    text_layer: str = ""             # embedded text (PDF only)
    text_spans: list[dict[str, Any]] = field(default_factory=list)  # PDF spans with bbox/font
    drawings: list[dict[str, Any]] = field(default_factory=list)    # PDF vector fills
    width_pt: float | None = None
    height_pt: float | None = None


@dataclass
class Document:
    filename: str
    kind: str                         # "pdf" | "png" | "jpeg"
    data: bytes
    pages: list[Page]
    metadata: dict[str, Any] = field(default_factory=dict)
    exif: dict[str, Any] = field(default_factory=dict)
    total_pages: int = 0
    truncated: bool = False

    @property
    def is_pdf(self) -> bool:
        return self.kind == "pdf"


def sniff_kind(data: bytes) -> str | None:
    if data.startswith(b"%PDF-") or data[:1024].find(b"%PDF-") != -1:
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return None


def validate_upload(filename: str, data: bytes) -> str:
    """Return the detected kind or raise ``DocumentError``."""
    if not filename:
        raise DocumentError("no_file", "No file was selected.")
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentError("unsupported_type",
                            "Unsupported file type. Please upload a PDF, PNG, JPG or JPEG file.")
    if len(data) == 0:
        raise DocumentError("empty_file", "The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise DocumentError("too_large", f"File exceeds the {MAX_FILE_BYTES // (1024*1024)} MB limit.")
    kind = sniff_kind(data)
    if kind is None:
        raise DocumentError("unrecognised_content",
                            "The file content is not a valid PDF, PNG or JPEG.")
    expected = ALLOWED_EXTENSIONS[ext]
    if kind != expected:
        raise DocumentError("extension_mismatch",
                            f"The file extension says {expected.upper()} but the content is {kind.upper()}.")
    return kind


def _load_image(filename: str, kind: str, data: bytes) -> Document:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise DocumentError("corrupted", f"The image could not be decoded ({exc.__class__.__name__}).") from exc
    exif: dict[str, Any] = {}
    try:
        from PIL.ExifTags import TAGS
        raw = img.getexif()
        for k, v in raw.items():
            name = TAGS.get(k, str(k))
            if isinstance(v, bytes):
                continue
            exif[name] = str(v)[:200]
    except Exception:
        pass
    img = ImageOps.exif_transpose(img).convert("RGB")
    info = {"format": kind.upper(), "mode": img.mode, "size": img.size}
    dpi = None
    try:
        dpi = Image.open(io.BytesIO(data)).info.get("dpi")
    except Exception:
        pass
    if dpi:
        info["dpi"] = tuple(float(x) for x in dpi)
    return Document(filename=filename, kind=kind, data=data,
                    pages=[Page(number=1, image=img)], metadata=info, exif=exif, total_pages=1)


def _load_pdf(filename: str, data: bytes) -> Document:
    import pymupdf as fitz

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentError("corrupted", "The PDF could not be opened; it may be corrupted.") from exc
    if doc.needs_pass:
        raise DocumentError("encrypted", "The PDF is password-protected and cannot be analysed.")
    if doc.page_count == 0:
        raise DocumentError("corrupted", "The PDF contains no pages.")

    pages: list[Page] = []
    n = min(doc.page_count, MAX_PDF_PAGES)
    try:
        for i in range(n):
            p = doc.load_page(i)
            pix = p.get_pixmap(dpi=RENDER_DPI, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            text = p.get_text("text") or ""
            spans: list[dict[str, Any]] = []
            for block in p.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for s in line.get("spans", []):
                        if s.get("text", "").strip():
                            spans.append({"text": s["text"], "bbox": tuple(s["bbox"]),
                                          "font": s.get("font", ""), "size": round(s.get("size", 0), 1),
                                          "color": s.get("color", 0)})
            drawings: list[dict[str, Any]] = []
            try:
                for d in p.get_drawings():
                    if d.get("fill") is not None:
                        drawings.append({"rect": tuple(d["rect"]), "fill": tuple(d["fill"]),
                                         "fill_opacity": d.get("fill_opacity", 1.0)})
            except Exception:
                pass
            pages.append(Page(number=i + 1, image=img, text_layer=text, text_spans=spans,
                              drawings=drawings, width_pt=p.rect.width, height_pt=p.rect.height))
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("corrupted", "A PDF page could not be rendered; the file may be damaged.") from exc

    meta = {k: v for k, v in (doc.metadata or {}).items() if v}
    meta["page_count"] = doc.page_count
    meta["pdf_version"] = data[:8].decode("latin-1", "replace").strip()
    result = Document(filename=filename, kind="pdf", data=data, pages=pages, metadata=meta,
                      total_pages=doc.page_count, truncated=doc.page_count > MAX_PDF_PAGES)
    doc.close()
    return result


def load_document(filename: str, data: bytes) -> Document:
    kind = validate_upload(filename, data)
    if kind == "pdf":
        return _load_pdf(filename, data)
    return _load_image(filename, kind, data)
