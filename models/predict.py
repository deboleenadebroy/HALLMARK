"""Inference for the visual forgery classifier.

The model is loaded once, lazily, from ``HALLMARK_MODEL_DIR`` (default
``models/artifacts``). If PyTorch is missing or no trained model is present,
``analyse_visual`` returns a NOT_PERFORMED evidence item — Hallmark never
pretends to run AI classification it cannot run.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from services.evidence import Evidence, Finding, Status, Strength

DEFAULT_DIR = Path(__file__).resolve().parent / "artifacts"
_lock = threading.Lock()
_cache: dict[str, Any] = {}


def model_dir() -> Path:
    return Path(os.environ.get("HALLMARK_MODEL_DIR", DEFAULT_DIR))


class VisualModel:
    def __init__(self, directory: Path):
        import torch
        from models.train import build_model

        self.dir = directory
        self.cfg = json.loads((directory / "model_config.json").read_text())
        self.model = build_model(self.cfg["arch"], pretrained=False)
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.T = float(self.cfg.get("temperature", 1.0))
        self.calibrated = "temperature" in self.cfg
        ood = self.cfg.get("ood") or {}
        self.ood = None
        if ood.get("file") and (directory / ood["file"]).exists():
            z = np.load(directory / ood["file"])
            self.ood = {"mean": z["mean"], "precision": z["precision"],
                        "p95": float(ood["val_p95"]), "p99": float(ood["val_p99"])}
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    def predict(self, image) -> dict[str, Any]:
        import torch
        from models.train import features_and_logits, make_transforms

        x = make_transforms(self.cfg["img_size"], train=False)(image).unsqueeze(0)
        with torch.no_grad():
            feats, logits = features_and_logits(self.model, x)
        lg = logits.numpy()[0].astype(np.float64)
        z = lg / self.T
        z -= z.max()
        p = np.exp(z) / np.exp(z).sum()
        out = {"p_forged": float(p[1]), "p_authentic": float(p[0]),
               "logit_margin": float((lg[1] - lg[0]) / self.T)}
        if self.ood:
            d = feats.numpy()[0] - self.ood["mean"]
            dist = float(np.sqrt(d @ self.ood["precision"] @ d))
            out["ood_distance"] = dist
            out["ood"] = "out_of_distribution" if dist > self.ood["p99"] else (
                "borderline" if dist > self.ood["p95"] else "in_distribution")
        else:
            out["ood"] = "unknown"
        return out


def load_model() -> tuple[VisualModel | None, str | None]:
    d = model_dir()
    key = str(d)
    with _lock:
        if key in _cache:
            return _cache[key]
        try:
            import torch  # noqa: F401
        except Exception:
            res = (None, "PyTorch is not installed, so the visual model cannot run.")
        else:
            if not (d / "model.pt").exists() or not (d / "model_config.json").exists():
                res = (None, "No trained visual model is installed (models/artifacts/model.pt is missing).")
            else:
                try:
                    res = (VisualModel(d), None)
                except Exception as exc:
                    res = (None, f"The visual model could not be loaded: {exc.__class__.__name__}: {exc}"[:300])
        _cache[key] = res
        return res


def reset_cache() -> None:
    with _lock:
        _cache.clear()


def validation_status(cfg: dict) -> tuple[bool, str | None]:
    """A model is used only if its held-out evaluation passed the validation gate."""
    v = cfg.get("validation")
    if v is None:
        return False, "the installed model has no recorded held-out validation"
    if v.get("passed"):
        return True, None
    return False, "the installed model failed held-out validation (" + "; ".join(v.get("reasons", [])) + ")"


def _gate_evidence(cfg: dict, why: str) -> Evidence:
    m = cfg.get("metrics", {}).get("test_calibrated", {})
    return Evidence("visual_model", Status.NOT_PERFORMED,
                    "AI visual classification was not used: " + why + ". Its predictions would not be reliable.",
                    limitations=["Train on a dataset with more independent genuine and forged certificates "
                                 "(see docs/EVALUATION.md)."],
                    details={"validation": cfg.get("validation"),
                             "test_metrics": {k: m.get(k) for k in ("accuracy", "balanced_accuracy", "roc_auc", "n")},
                             "evaluation": cfg.get("metrics", {}).get("evaluation")})


def analyse_visual(pages) -> Evidence:
    override = os.environ.get("HALLMARK_ALLOW_UNVALIDATED_MODEL") == "1"
    cfg_path = model_dir() / "model_config.json"
    if cfg_path.exists() and not override:
        # Check the recorded validation first: a model that failed is never loaded, so no PyTorch is needed.
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = None
        if cfg is not None:
            ok, why = validation_status(cfg)
            if not ok:
                return _gate_evidence(cfg, why)
    model, reason = load_model()
    if model is None:
        return Evidence("visual_model", Status.NOT_PERFORMED,
                        "AI visual classification was not performed. " + reason,
                        limitations=["See README → 'Training the visual model' to train and install it."])
    ok, why = validation_status(model.cfg)
    if not ok and not override:
        return _gate_evidence(model.cfg, why)
    preds = []
    for p in pages:
        r = model.predict(p.image)
        r["page"] = p.number
        preds.append(r)
    # document-level: the most suspicious in-distribution page drives the verdict
    usable = [r for r in preds if r["ood"] in ("in_distribution", "borderline", "unknown")]
    m = model.cfg.get("metrics", {}).get("test_calibrated", {})
    limitations = [
        f"Trained on {model.cfg.get('dataset', {}).get('name', 'the training dataset')}; "
        "certificate designs unlike the training data may be misclassified.",
        "The model judges visual appearance only; it cannot confirm that an institution issued the certificate.",
    ]
    if not model.calibrated:
        limitations.append("Model probabilities are not calibrated.")
    if not ok:
        limitations.insert(0, "RESEARCH OVERRIDE: this model failed validation (" + why + ") and is "
                              "used only because HALLMARK_ALLOW_UNVALIDATED_MODEL=1.")
    details = {"pages": preds, "temperature": model.T, "calibrated": model.calibrated,
               "test_metrics": {k: m.get(k) for k in ("accuracy", "precision_forged", "recall_forged",
                                                      "f1_forged", "roc_auc", "brier", "ece_15", "n")},
               "arch": model.cfg.get("arch")}
    if not usable:
        return Evidence("visual_model", Status.INCONCLUSIVE,
                        "The certificate looks unlike the images the model was trained on, so its visual "
                        "prediction is not relied on.",
                        findings=[Finding("visual_out_of_distribution",
                                          "All pages were flagged as out-of-distribution for the visual model.",
                                          0, Strength.NONE)],
                        limitations=limitations, details=details)
    worst = max(usable, key=lambda r: r["p_forged"])
    details["document_p_forged"] = worst["p_forged"]
    details["document_ood"] = worst["ood"]
    pf = worst["p_forged"]
    note = " (borderline match to training data)" if worst["ood"] == "borderline" else ""
    if pf >= 0.5:
        st, msg, direction = Status.FAIL, f"The visual model estimates a {pf:.0%} chance of forgery{note}.", -1
    else:
        st, msg, direction = Status.PASS, f"The visual model estimates a {1 - pf:.0%} chance the design is authentic{note}.", +1
    strength = Strength.MODERATE if abs(pf - 0.5) >= 0.3 else Strength.WEAK
    return Evidence("visual_model", st, msg,
                    findings=[Finding("visual_prediction", msg, direction, strength, page=worst["page"],
                                      data={"p_forged": pf, "ood": worst["ood"]})],
                    limitations=limitations, details=details)
