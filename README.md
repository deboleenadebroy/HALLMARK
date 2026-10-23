# Hallmark

*Your mark of authenticity.*

Hallmark assesses whether an uploaded certificate (PDF, PNG, JPG or JPEG) is likely to be
genuine. It combines embedded digital-signature verification, document-integrity forensics,
OCR-based consistency checks and — once trained — an AI visual forgery model, then reports an
**Estimated Authenticity Score** with a plain-language explanation and a downloadable report.

No certificate ID, URL or key is ever requested, and analysis runs entirely offline.

> Hallmark gives the most defensible estimate the available evidence supports. It is not a legal
> verification. Only the issuing institution can confirm that a certificate was issued.

---

## Quick start

```bash
# 1. System dependency: Tesseract OCR
#    Ubuntu/Debian: sudo apt install tesseract-ocr
#    macOS:         brew install tesseract
#    Windows:       https://github.com/UB-Mannheim/tesseract/wiki  (add to PATH)

# 2. Python 3.10+
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-ml.txt                       # optional: AI visual model (PyTorch)

# 3. Run
python app.py                                            # → http://127.0.0.1:5000
```

Optional settings live in `.env` (copy `.env.example`). Google Fonts (Playfair Display, Poppins)
load when online; the interface falls back to system fonts offline.

## Demo

See **DEMO.md**. Quick start: `bash run_demo.sh` (macOS/Linux) or `run_demo.bat` (Windows), then open http://127.0.0.1:8000.

```bash
python scripts/make_demo_files.py --trust     # creates demo/ and trusts the demo signing CA
python app.py                                 # http://127.0.0.1:5000, then upload the files in demo/
```

| File | What it shows | Result |
|---|---|---|
| `1_signed_certificate.pdf` | valid signature chained to a trusted CA | ~95% · Strong evidence of authenticity (VERIFIED) |
| `2_altered_after_signing.pdf` | bytes changed inside the signed range | ~6% · Likely forged |
| `3_content_added_after_signing.pdf` | new content appended after signing | ~9% · Likely forged |
| `4_name_covered_and_retyped.pdf` | white box over the name, new name typed on top | ~26% · Suspicious |
| `5_clean_scan.jpg` | unsigned scan, no issues | ~60% · Inconclusive (heuristics alone cannot verify) |
| `6_name_pasted_into_scan.jpg` | clean name patch pasted into a scan | ~46% · Inconclusive, heat-map in details |
| `7_future_dated_certificate.pdf` | issue date in the future | ~35% · Suspicious |
| `8_not_a_certificate.pdf` | a sales memo | score withheld (out of scope) |

All demo documents and keys are synthetic. Remove `trust_anchors/demo_root_ca.pem` after the demo.

## What the user sees

1. Upload one certificate and click **Analyse Certificate**.
2. A score ring with one of five labels, a short explanation and three key findings.
3. **Detailed analysis** (expandable): AI visual model, cryptographic signature, document
   integrity, extracted details, checks not performed, how the score was formed, file fingerprint.
4. **Download Report** — a self-contained HTML report (prints cleanly to PDF).

| Score | Label | Meaning |
|---|---|---|
| 90–100 | Strong evidence of authenticity | Strong evidence consistent with authenticity |
| 70–89 | Likely authentic | Evidence favours authenticity, uncertainty remains |
| 40–69 | Inconclusive | Additional review recommended |
| 20–39 | Suspicious | Significant suspicious evidence |
| 0–19 | Likely forged | Strong evidence consistent with forgery |
| — | Inconclusive — score unavailable | Unreadable or out-of-scope document; no number is invented |

These bands are **product defaults, not validated decision thresholds**. Re-check them against
held-out real data before treating them as decision rules.

## How the assessment works

```
upload ─► validation (magic bytes, size, type) ─► PDF render / image load ─► OCR (deskew, rotate, CLAHE)
            │
            ├─ cryptography        embedded PDF signatures: digest, RSA validity, padding/hash policy,
            │                      X.509 signer, trust anchors, modifications after signing
            ├─ document_integrity  covered/hidden text, white overlays, mixed fonts, metadata,
            │                      incremental saves; image pixel-noise splice check
            ├─ identity_and_details field-aware extraction, date/ID/placeholder consistency
            ├─ visual_model        calibrated CNN + out-of-distribution detection (if trained)
            ├─ issuer_discovery    NOT_PERFORMED (offline; apparent issuer shown only)
            └─ registry            NOT_PERFORMED (no external lookups)
                       │
                       ▼
     authenticity_scorer (log-odds) ─► tier6_classifier (6 states) ─► UI / report
```

Every check reports `PASS`, `FAIL`, `INCONCLUSIVE`, `NOT_APPLICABLE` or `NOT_PERFORMED`, with an
explanation, supporting findings and limitations. A missing signature is *not applicable*, not a
failure; an OCR failure is *inconclusive*, not evidence of forgery.

### The score (and what it is not)

The score is a **documented heuristic** (`services/authenticity_scorer.py`), not a calibrated
probability. Evidence is added in log-odds space from a neutral 50%:

| Evidence | Contribution |
|---|---|
| Valid signature chained to a configured trust anchor | +2.5 |
| Valid signature, signer not trusted | +0.4 (anyone can self-sign) |
| Signed content altered / invalid signature | −3.0 |
| Changed after signing | −2.5 |
| Visual model (calibrated logit, clipped ±3; halved if borderline OOD; 0 if OOD) | −3 … +3 |
| Moderate integrity / detail findings (covered text, placeholders, contradictions) | −0.8 … −0.9 each |
| Weak heuristic indicators | −0.3 … −0.35 each, **−0.9 total cap** |
| No integrity issues / consistent details | +0.2 each |

Consequences, enforced by tests:

* Heuristics alone stay **below ~60%**. Only a trusted signature can reach the top band, and
  only a trusted signature can produce `VERIFIED`. Visual evidence alone never verifies.
* Weak indicators alone cannot push a score below ~29%.
* Certificate IDs are extracted for display but **never** affect the score.
* Strong positive and strong negative evidence together → conflict flagged, total halved toward 50%.
* Unreadable or non-certificate documents → score withheld (unless a signature or the model gives
  decisive evidence).

Only the visual-model component is calibrated (temperature scaling on the validation split).
If you collect a labelled set of real genuine/forged certificates, fit a calibrator (e.g.
logistic regression on the evidence features) on it before calling the output a probability.

## Training the visual model

```bash
pip install -r requirements-ml.txt
python models/train.py --data-dir <dataset> --inspect-only     # labels, duplicates, independent sources, split leakage
python models/train.py --data-dir <dataset> --out models/artifacts
```

`train.py` reads class folders, a labels CSV, or a **YOLO / Roboflow export** (`data.yaml`, `images/`, `labels/`;
an image is "forged" if any labelled region is a forged class). It recovers the independent **source documents**
(Roboflow file names, otherwise perceptual-hash grouping) and evaluates only across them, never across copies.

* **Few sources (< 50) → `linear_probe`:** frozen ImageNet EfficientNet-B0 plus regularised logistic regression,
  nested leave-one-source-out evaluation and calibration, out-of-distribution statistics, plus a region-crop diagnostic
  for YOLO data. The backbone weights are fetched from GitHub with a hash check (or `--pretrained-weights <file>`).
* **Many sources → `finetune`:** grouped 70/15/15 split, augmentation on training only, EfficientNet-B0/MobileNetV3
  fine-tuning, temperature scaling, and test metrics (accuracy, precision, recall, F1, confusion matrix,
  ROC-AUC, Brier score, ECE). On a GPU, use `notebooks/kaggle_train.ipynb`.

**Validation gate.** The app uses a model only if its held-out-source ROC-AUC is ≥ 0.80, its balanced accuracy is
≥ 0.70, and each class has ≥ 3 independent sources. Otherwise the AI check is shown as *not used*, with the reason.

**Result on the Kaggle Certificate Forgery Dataset:** trained, **failed validation, not used.** The dataset holds
254 images but only 15 source documents, all one student's VTU grade card. Its official split scores 100% only
because every test document also appears in training. Held out by source, the whole-image model reaches ROC-AUC 0.65 and balanced accuracy 0.43: it calls every
genuine page forged.
Details and per-source results: `docs/EVALUATION.md`. The trained artifacts are in `models/artifacts/`
(git-ignored; about 27 MB).

## Digital signatures and trust

Embedded PDF signatures are checked automatically with pyHanko. To let Hallmark call a signature
*trusted*, place the signing authority's root certificate in `trust_anchors/`. Revocation
(CRL/OCSP) is not checked, because analysis is offline. A valid signature proves the bytes are
unchanged since signing; it does not prove the signer is a legitimate institution.

`services/certificate_verifier.verify_rsa_signature` is a strict standalone RSA verifier
(PKCS#1 v1.5 or PSS; SHA-256/384/512; ≥ 2048-bit keys; MD5/SHA-1 rejected).

## Project structure

```
app.py                       Flask app: upload, result, report download, JSON API (/api/analyse)
services/
  evidence.py                Status enum, Evidence/Finding data model, categories
  document_reader.py         upload validation, PDF text/render (PyMuPDF), image loading
  ocr_service.py             Tesseract OCR with rotation, deskew, contrast, scaling
  certificate_parser.py      field-aware extraction (names, issuer, dates, labelled IDs), consistency
  content_analyzer.py        identity_and_details / issuer_discovery / registry evidence
  certificate_verifier.py    RSA verification, PDF signatures, X.509, trust anchors
  security_analyzer.py       SHA-256, PDF revisions, active content
  integrity_analyzer.py      PDF and image integrity heuristics
  authenticity_scorer.py     Estimated Authenticity Score
  tier6_classifier.py        six-state classification, explanation, key findings, recommendations
  report_generator.py        self-contained HTML report
  pipeline.py                orchestration
models/  train.py · linear_probe.py · predict.py · artifacts/
templates/ static/ assets/   minimal Hallmark UI and logo
tests/                       91 unit + integration tests, fixtures, synthetic dataset builder
scripts/evaluate_heuristics.py   reproducible heuristic evaluation → docs/heuristics_eval.json
notebooks/kaggle_train.ipynb     training on Kaggle
docs/EVALUATION.md               what has and has not been evaluated
```

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

The suite covers PDF, PNG, JPG and JPEG uploads; multi-page, corrupted, encrypted and unsupported
files; rotated, skewed and low-quality scans; the `IFICATE` OCR-fragment regression; valid,
invalid, PSS, weak-hash and short-key RSA cases; trusted, untrusted, tampered and modified-after-signing
PDF signatures; unsupported signature formats; missing trust anchors; covered text, white overlays,
incremental saves and splice detection; scoring edge cases, conflicts and band boundaries;
missing-model behaviour; a training/inference smoke test on synthetic data (skipped without
PyTorch); report generation; and the web routes end to end. All test keys are throwaway keys
generated during the run.

## Privacy and security

* Hallmark never saves uploaded files: they are analysed in memory (Werkzeug may buffer a large upload
  in a temporary file that is deleted when the request ends). Results are kept in memory for
  30 minutes so the report can be downloaded, then discarded.
* Uploaded content is never executed. Embedded URLs are not fetched. PDF JavaScript and actions are
  reported, never run.
* No secrets are committed: configuration is read from environment variables or `.env` (git-ignored).
* For deployment, put Hallmark behind a production WSGI server (e.g. `gunicorn -w 2 app:app`) and HTTPS.

## Known limitations

* The only available training dataset does not support a usable visual model, so the AI check is currently off (see above).
* The integrity heuristics are validated only on synthetic data (`docs/EVALUATION.md`). They can
  miss skilled edits and can flag unusual but legitimate designs.
* The pixel-noise check cannot see edits on noiseless, born-digital images.
* OCR can misread names, dates and codes. Non-English certificates are only partly supported
  (Tesseract `eng` by default).
* Issuer and registry verification are not performed. Hallmark is offline by design.
