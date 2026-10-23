# Evaluation record

This file records what has actually been measured. Nothing here is an estimate of real-world
accuracy unless it says so explicitly.

## 1. Visual forgery model — trained, **failed validation, not used**

Trained on 2026-10-23 with `python models/train.py --data-dir <dataset>` (method chosen automatically:
`linear_probe`). Full records: `models/artifacts/data_report.json`, `evaluation.json`,
`oof_predictions.json`, `model_config.json`.

### What the dataset actually is

| Item | Finding |
|---|---|
| Source | Kaggle `harikrish03/certificate-forgery-dataset` = Roboflow `certificate-forgery-detection-avxar` v1 (CC BY 4.0) |
| Format | YOLOv8 object detection, not image classification: 3 labelled regions per image (university logo, seal, registrar signature), each `fake` or `true` |
| Images | 254 (112 genuine, 142 forged; an image is "forged" if any region is `fake`), all resized to 640×640 |
| Independent source documents | **15**: 3 genuine pages and 12 edited versions of them, recovered from Roboflow file names |
| Exact duplicate files | 239 |
| Content | Every image is the **same student's VTU grade card**; forgeries replace or smear the logo, seal or signature |
| Official split leakage | **12 of 12** test source documents also appear in train/val |

### Method

Frozen ImageNet EfficientNet-B0 (original weights, SHA-256-verified) → global-average features at 448 px →
L2-regularised logistic regression. Evaluation: **nested leave-one-source-out**. Each source document is held out
in turn, the regularisation strength is chosen by an inner leave-one-source-out loop on the remaining sources,
and calibration (temperature scaling) is also fitted without the held-out source.
Fine-tuning was not used because 15 source documents cannot support it.

### Results

| Evaluation | n | Accuracy | Balanced acc. | ROC-AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Held-out source, uncalibrated | 254 | 0.484 | 0.433 | 0.651 | 0.308 | 0.406 |
| Held-out source, calibrated | 254 | 0.484 | 0.433 | 0.286* | 0.282 | 0.355 |
| Dataset's official split (leaky) | 40 | 1.000 | — | 1.000 | 0.017 | — |
| *Diagnostic:* labelled region crops, held-out source | 762 regions | 0.982 | — | 0.996 | — | — |

\* Nested calibration fits a separate temperature for each held-out source, which distorts rankings *across*
sources, so the calibrated ROC-AUC is not a fair ranking measure. The gate therefore uses the uncalibrated
held-out ROC-AUC (**0.651**) for ranking and the calibrated probabilities for balanced accuracy.

Held-out-source confusion matrix (calibrated; rows = true authentic/forged, columns = predicted):
`[[0, 112], [19, 123]]`. Precision (forged) 0.523, recall (forged) 0.866,
F1 0.653. Fitted temperature 20.0 (the upper limit), which flattens predictions to about 50%:
the whole-image model carries essentially no reliable signal.

Per held-out source document:

| Source | Label | Images | Mean P(forged) | Accuracy |
|---|---|---|---|---|
| `Upto-4th-Sem-Markscard_page-0002` | authentic | 38 | 0.569 | 0.00 |
| `Upto-4th-Sem-Markscard_page-0003` | authentic | 37 | 0.505 | 0.00 |
| `Upto-4th-Sem-Markscard_page-0004` | authentic | 37 | 0.551 | 0.00 |
| `f1` | forged | 13 | 0.527 | 1.00 |
| `f10` | forged | 13 | 0.524 | 1.00 |
| `f2` | forged | 13 | 0.508 | 1.00 |
| `f3` | forged | 13 | 0.192 | 0.00 |
| `f4` | forged | 13 | 0.521 | 1.00 |
| `f5` | forged | 13 | 0.507 | 1.00 |
| `f6` | forged | 13 | 0.516 | 1.00 |
| `f7` | forged | 13 | 0.525 | 1.00 |
| `f8` | forged | 13 | 0.515 | 1.00 |
| `f9` | forged | 13 | 0.506 | 1.00 |
| `fnew1` | forged | 6 | 0.492 | 0.00 |
| `fnew2` | forged | 6 | 0.524 | 1.00 |

### Interpretation

* **The 100% official-split accuracy is entirely leakage.** Copies of each test document are in training.
* **The whole-image model does not generalise, even within this one grade card.** Every held-out genuine page
  is called forged. The forgeries are edits of the same three pages, so when a genuine page is held out, its
  forged copies remain in training and the model learns "this page's content = forged". Below-chance balanced
  accuracy (0.43), with 0 of 112 genuine images correct, is the signature of this confound.
* **The edits are visually detectable where they are.** Region crops (logo/seal/signature) separate almost
  perfectly across held-out sources. But the genuine regions all come from one university and one registrar, so
  this shows "differs from *this* card's seal" rather than general forgery detection. It would also need a
  region detector, which this data (one template) cannot train for general certificates.

### Decision

The model failed the validation gate (`models/linear_probe.py: VALIDATION_GATE`: held-out-source ROC-AUC ≥ 0.80,
balanced accuracy ≥ 0.70, ≥ 3 independent sources per class). It failed on ROC-AUC 0.651 and balanced accuracy 0.433. Hallmark therefore reports the AI check as
**not used, with the reason**, and the score is unaffected. Setting `HALLMARK_ALLOW_UNVALIDATED_MODEL=1` enables it
for research only, with a warning in every result.

### What would make a usable model

A dataset of **many independent certificates** from different issuers and layouts (genuine and forged, ideally
forgeries of *different* documents than the genuine ones). The same command then trains, evaluates by source,
and is enabled automatically if it passes the gate.

## 2. Integrity heuristics — synthetic controlled evaluation

Reproduce with `python scripts/evaluate_heuristics.py --n 40` (writes `docs/heuristics_eval.json`).

Setup: synthetic certificate images rendered with random recipients, Gaussian "scan" noise of
the given σ (0–255 scale) and JPEG quality 75/85/92. "Spliced" images have a clean, digitally
rendered name patch pasted over the name. PDFs are generated with ReportLab; "covered" PDFs have a
white rectangle drawn over the original name and a new name typed on top.

| Check | Condition | False positives (genuine) | Detected (manipulated) |
|---|---|---|---|
| Pixel-noise inconsistency | σ = 2 | 0 / 40 | 27 / 40 |
| Pixel-noise inconsistency | σ = 3 | 0 / 40 | 40 / 40 |
| Pixel-noise inconsistency | σ = 5 | 0 / 40 | 40 / 40 |
| Pixel-noise inconsistency | σ = 8 | 0 / 40 | 40 / 40 |
| PDF covered text | — | 0 / 20 | 19 / 20 |

Interpretation and caveats:

* These are easy, synthetic manipulations. Real forgeries (matched noise, re-scanned prints,
  careful retouching) will be detected far less often, and real genuine certificates
  (photos, gradients, security patterns) may trigger false positives that this data does not contain.
* Because of this, both checks keep small weights in the score (weak/moderate), and heuristics
  alone cannot move a score below ~29% or above ~60% unless a moderate finding is present.
* The noise check is disabled for noiseless born-digital images, where it cannot work.

### Dropped during development

**Error-level analysis (ELA)** was implemented and measured on the same kind of synthetic data
(16 genuine vs 16 spliced JPEGs): it flagged 7/16 genuine and 9/16 spliced images. That is
no meaningful separation, so it was removed instead of shipped. A first noise-variance version
that compared all blocks, rather than flat background blocks only, failed in the same way and
was replaced by the current method.

## 3. Score bands

The five score bands (90/70/40/20) are product defaults. They have **not** been validated on real
labelled data. Once a labelled set of real genuine and forged certificates exists, measure the
score distribution per class and adjust the bands, or replace the heuristic combination with a
fitted, calibrated model.
