"""Hallmark — AI-assisted certificate authenticity assessment.

Run:  python app.py           (http://127.0.0.1:5000)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

from flask import (Flask, Response, abort, jsonify, render_template, request, send_from_directory)
from werkzeug.exceptions import RequestEntityTooLarge

from services.document_reader import MAX_FILE_BYTES, DocumentError
from services.pipeline import analyse_bytes
from services.report_generator import render_report

BASE = Path(__file__).resolve().parent

try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except Exception:
    pass
RESULT_TTL_S = 30 * 60
RESULT_MAX = 50

log = logging.getLogger("hallmark")


class ResultStore:
    """In-memory store so the report can be downloaded. Uploaded files are never stored."""

    def __init__(self, ttl=RESULT_TTL_S, maxlen=RESULT_MAX):
        self.ttl, self.maxlen = ttl, maxlen
        self._d: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, result: dict) -> None:
        with self._lock:
            self._purge()
            self._d[result["id"]] = (time.time(), result)
            while len(self._d) > self.maxlen:
                self._d.popitem(last=False)

    def get(self, rid: str) -> dict | None:
        with self._lock:
            self._purge()
            item = self._d.get(rid)
            return item[1] if item else None

    def _purge(self) -> None:
        now = time.time()
        for k in [k for k, (t, _) in self._d.items() if now - t > self.ttl]:
            del self._d[k]


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES + 512 * 1024
    app.config["SECRET_KEY"] = os.environ.get("HALLMARK_SECRET_KEY", os.urandom(32))
    store = ResultStore()
    app.extensions["hallmark_store"] = store

    def _analyse_upload():
        f = request.files.get("certificate")
        if f is None or not f.filename:
            raise DocumentError("no_file", "No file was selected.")
        data = f.read(MAX_FILE_BYTES + 1)
        name = os.path.basename(f.filename)[:200]
        result = analyse_bytes(name, data)
        store.put(result)
        return result

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=MAX_FILE_BYTES // (1024 * 1024))

    @app.post("/analyse")
    def analyse():
        try:
            result = _analyse_upload()
        except DocumentError as e:
            return render_template("index.html", error=e.message,
                                   max_mb=MAX_FILE_BYTES // (1024 * 1024)), 400
        except RequestEntityTooLarge:
            raise
        except Exception:
            log.exception("analysis failed")
            return render_template("index.html", error="The analysis failed unexpectedly. Please try another file.",
                                   max_mb=MAX_FILE_BYTES // (1024 * 1024)), 500
        return render_template("result.html", r=result)

    @app.post("/api/analyse")
    def api_analyse():
        try:
            result = _analyse_upload()
        except DocumentError as e:
            return jsonify({"error": e.code, "message": e.message}), 400
        for ev in result["evidence"]:
            ev.get("details", {}).pop("heatmap_png_b64", None)
        return jsonify(result)

    @app.get("/report/<rid>")
    def report(rid: str):
        if not rid.isalnum():
            abort(404)
        result = store.get(rid)
        if result is None:
            return render_template("index.html", error="That report has expired. Please analyse the certificate again.",
                                   max_mb=MAX_FILE_BYTES // (1024 * 1024)), 404
        html = render_report(result)
        stem = Path(result["filename"]).stem[:60] or "certificate"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
        return Response(html, mimetype="text/html",
                        headers={"Content-Disposition": f'attachment; filename="hallmark-report-{safe}.html"'})

    @app.get("/assets/<path:name>")
    def assets(name: str):
        return send_from_directory(BASE / "assets", name)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        msg = f"File exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit."
        if request.path.startswith("/api/"):
            return jsonify({"error": "too_large", "message": msg}), 413
        return render_template("index.html", error=msg, max_mb=MAX_FILE_BYTES // (1024 * 1024)), 413

    @app.after_request
    def headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        return resp

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host=os.environ.get("HALLMARK_HOST", "127.0.0.1"), port=int(os.environ.get("HALLMARK_PORT", "5000")),
            debug=os.environ.get("HALLMARK_DEBUG") == "1")
