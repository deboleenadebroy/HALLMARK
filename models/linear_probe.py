"""Small-data training path: frozen ImageNet backbone + regularised linear probe.

Used automatically when a dataset has few independent source documents (the
Kaggle certificate-forgery dataset has 254 images but only 15 source
documents). Fine-tuning a CNN on so few sources mostly memorises them; a
linear probe on frozen features has far fewer free parameters.

Evaluation is **leave-one-source-out**: every image is scored by a model that
never saw any copy of its source document. The regularisation strength is
chosen by an inner leave-one-source-out loop (nested), and calibration is
evaluated with a nested scheme too, so none of the reported numbers are fitted
on the data they are measured on.

Backbone: EfficientNet-B0 (``efficientnet_pytorch`` reference implementation,
Apache-2.0) with the original ImageNet weights, downloaded from the project's
GitHub release and verified against the SHA-256 prefix in the file name.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

EFFNET_URL = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
FEATURE_DIM = 1280
C_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)

# A model is only used by the app if its held-out-source evaluation clears this bar.
VALIDATION_GATE = {"min_roc_auc": 0.80, "min_balanced_accuracy": 0.70, "min_sources_per_class": 3}


# --------------------------------------------------------------------------- model

def _torch():
    import torch
    return torch


def build_backbone(pretrained: bool = True, weights_path: str | None = None):
    torch = _torch()
    from efficientnet_pytorch import EfficientNet
    m = EfficientNet.from_name("efficientnet-b0")
    if pretrained:
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        else:
            sd = torch.hub.load_state_dict_from_url(EFFNET_URL, map_location="cpu", check_hash=True, progress=False)
        m.load_state_dict(sd)
    m.eval()
    return m


def make_probe_module(backbone=None):
    """nn.Module: backbone → global-average features → standardise → linear logit."""
    torch = _torch()
    nn = torch.nn

    class LinearProbe(nn.Module):
        is_linear_probe = True

        def __init__(self, bb):
            super().__init__()
            self.backbone = bb
            self.register_buffer("mu", torch.zeros(FEATURE_DIM))
            self.register_buffer("sd", torch.ones(FEATURE_DIM))
            self.head = nn.Linear(FEATURE_DIM, 1)

        def features(self, x):
            return self.backbone.extract_features(x).mean((2, 3))

        def forward(self, x):
            f = self.features(x)
            z = self.head((f - self.mu) / self.sd)
            return torch.cat([torch.zeros_like(z), z], 1)      # 2-class logits [authentic, forged]

    return LinearProbe(backbone if backbone is not None else build_backbone(pretrained=False))


def extract_features(items, img_size: int, backbone, batch: int = 16, log=print) -> np.ndarray:
    torch = _torch()
    from PIL import Image, ImageOps
    from models.train import make_transforms
    tf = make_transforms(img_size, train=False)
    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(items), batch):
            ims = []
            for it in items[i:i + batch]:
                with Image.open(it["path"]) as im:
                    ims.append(tf(ImageOps.exif_transpose(im).convert("RGB")))
            out.append(backbone.extract_features(torch.stack(ims)).mean((2, 3)).numpy())
    log(f"[probe] features for {len(items)} images in {time.time() - t0:.0f}s")
    return np.concatenate(out).astype(np.float64)


# --------------------------------------------------------------------------- fitting / CV

def _fit(X, y, C):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    clf.fit((X - mu) / sd, y)
    return mu, sd, clf


def _margin(model, X):
    mu, sd, clf = model
    return clf.decision_function((X - mu) / sd)


def _choose_C(X, y, g) -> float:
    """Inner leave-one-source-out choice of C by log loss."""
    from sklearn.metrics import log_loss
    groups = np.unique(g)
    best, best_c = np.inf, C_GRID[0]
    for C in C_GRID:
        p = np.zeros(len(y))
        for gg in groups:
            te = g == gg
            if len(np.unique(y[~te])) < 2:
                p[te] = y[~te].mean()
                continue
            p[te] = 1 / (1 + np.exp(-_margin(_fit(X[~te], y[~te], C), X[te])))
        ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
        if ll < best:
            best, best_c = ll, C
    return best_c


def _temperature(margins, y) -> float:
    from models.train import fit_temperature
    lg = np.stack([np.zeros_like(margins), margins], 1)
    return fit_temperature(lg, y)


def loso_evaluate(X, y, g, log=print) -> dict:
    """Nested leave-one-source-out: outer folds hold out one source document."""
    groups = np.unique(g)
    oof_margin = np.zeros(len(y))
    chosen = {}
    for k, gg in enumerate(groups):
        te = g == gg
        C = _choose_C(X[~te], y[~te], g[~te])
        chosen[str(gg)] = C
        oof_margin[te] = _margin(_fit(X[~te], y[~te], C), X[te])
        log(f"[probe] fold {k + 1}/{len(groups)} held-out source '{gg}' (n={te.sum()}, label={int(y[te].mean() >= .5)}) C={C}")
    # nested calibration: temperature for each held-out source fitted on the other sources' OOF margins
    cal = np.zeros(len(y))
    for gg in groups:
        te = g == gg
        T = _temperature(oof_margin[~te], y[~te])
        cal[te] = oof_margin[te] / T
    return {"oof_margin": oof_margin, "oof_margin_calibrated": cal, "C_per_fold": chosen}


def per_source(p, y, g) -> dict:
    out = {}
    for gg in np.unique(g):
        sel = g == gg
        out[str(gg)] = {"n": int(sel.sum()), "label": "forged" if y[sel].mean() >= .5 else "authentic",
                        "mean_p_forged": round(float(p[sel].mean()), 3),
                        "accuracy": round(float(((p[sel] >= .5) == y[sel]).mean()), 3)}
    return out


def gate(m: dict, y, g) -> dict:
    from sklearn.metrics import balanced_accuracy_score  # noqa: F401  (documented metric)
    srcs = {c: len(set(g[y == c])) for c in (0, 1)}
    bal = m.get("balanced_accuracy", 0.0)
    auc = m.get("roc_auc") or 0.0
    reasons = []
    if auc < VALIDATION_GATE["min_roc_auc"]:
        reasons.append(f"held-out-source ROC-AUC {auc:.3f} < {VALIDATION_GATE['min_roc_auc']}")
    if bal < VALIDATION_GATE["min_balanced_accuracy"]:
        reasons.append(f"held-out-source balanced accuracy {bal:.3f} < {VALIDATION_GATE['min_balanced_accuracy']}")
    if min(srcs.values()) < VALIDATION_GATE["min_sources_per_class"]:
        reasons.append(f"only {min(srcs.values())} independent source document(s) in the smaller class")
    return {"passed": not reasons, "reasons": reasons, "criteria": VALIDATION_GATE,
            "sources_per_class": {"authentic": srcs[0], "forged": srcs[1]}}


# --------------------------------------------------------------------------- region diagnostic (YOLO data)

def region_diagnostic(items, backbone, log=print) -> dict | None:
    """Leave-one-source-out probe on the labelled *regions* (box crops).

    Diagnostic only: it shows whether the manipulations are visually learnable
    where they occur. It is not usable at inference without a region detector.
    """
    torch = _torch()
    from PIL import Image
    from sklearn.metrics import roc_auc_score, accuracy_score
    from models.train import IMAGENET_MEAN, IMAGENET_STD
    boxes = [(it, b) for it in items for b in it.get("regions", [])]
    if len(boxes) < 30:
        return None
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    X, Y, G = [], [], []
    with torch.no_grad():
        for it in items:
            if not it.get("regions"):
                continue
            with Image.open(it["path"]) as im:
                im = im.convert("RGB")
                W, H = im.size
                xs = []
                for b in it["regions"]:
                    cx, cy, w, h = b["cx"] * W, b["cy"] * H, b["w"] * W, b["h"] * H
                    cr = im.crop((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)).resize((160, 160), Image.BICUBIC)
                    xs.append((torch.from_numpy(np.asarray(cr, dtype=np.float32) / 255).permute(2, 0, 1) - mean) / std)
                    Y.append(int(b["forged"]))
                    G.append(it["source"])
            X.append(backbone.extract_features(torch.stack(xs)).mean((2, 3)).numpy())
    X, Y, G = np.concatenate(X).astype(np.float64), np.array(Y), np.array(G)
    oof = np.zeros(len(Y))
    for gg in np.unique(G):
        te = G == gg
        oof[te] = 1 / (1 + np.exp(-_margin(_fit(X[~te], Y[~te], 0.001), X[te])))
    res = {"n_regions": int(len(Y)), "n_forged_regions": int(Y.sum()),
           "loso_roc_auc": round(float(roc_auc_score(Y, oof)), 4),
           "loso_accuracy": round(float(accuracy_score(Y, oof >= .5)), 4),
           "note": "Diagnostic only: regions come from ground-truth boxes; no detector is trained."}
    log(f"[probe] region diagnostic: {json.dumps(res)}")
    return res


# --------------------------------------------------------------------------- main entry

def run_probe(items, report, args, out_dir: Path, log=print) -> dict:
    torch = _torch()
    from models.train import metrics, softmax_pos, _plots, CLASSES

    torch.set_num_threads(max(1, min(8, __import__("os").cpu_count() or 1)))
    y = np.array([CLASSES.index(it["label"]) for it in items])
    g = np.array([it["source_group"] for it in items])
    log(f"[probe] {len(items)} images from {len(set(g))} independent source documents; "
        f"sources per class: {dict(Counter((it['label'] for it in {it['source_group']: it for it in items}.values())))}")

    backbone = build_backbone(pretrained=not args.no_pretrained, weights_path=args.pretrained_weights)
    X = extract_features(items, args.img_size, backbone, log=log)

    ev = loso_evaluate(X, y, g, log=log)
    p_raw = 1 / (1 + np.exp(-ev["oof_margin"]))
    p_cal = 1 / (1 + np.exp(-ev["oof_margin_calibrated"]))
    m_raw, m_cal = metrics(p_raw, y), metrics(p_cal, y)
    from sklearn.metrics import balanced_accuracy_score
    for m, p in ((m_raw, p_raw), (m_cal, p_cal)):
        m["balanced_accuracy"] = float(balanced_accuracy_score(y, p >= .5))
    _plots(out_dir, p_cal, y, "Held-out-source reliability (calibrated)")
    # ROC-AUC is a ranking metric: per-fold temperatures distort rankings *across* folds, so the gate uses
    # the uncalibrated held-out ranking for AUC and the calibrated probabilities for balanced accuracy.
    g_res = gate({**m_cal, "roc_auc": m_raw["roc_auc"]}, y, g)
    log(f"[probe] held-out-source (calibrated): " + json.dumps({k: m_cal[k] for k in (
        "accuracy", "balanced_accuracy", "precision_forged", "recall_forged", "f1_forged", "roc_auc", "brier", "ece_15")}))
    log(f"[probe] validation gate: {'PASSED' if g_res['passed'] else 'FAILED'} {g_res['reasons']}")

    # Comparison with the dataset's own split, if it has one (shows leakage effects).
    official = None
    splits = np.array([it.get("official_split") or "" for it in items])
    if {"train", "test"} <= set(splits):
        tr, te = splits != "test", splits == "test"
        mdl = _fit(X[tr], y[tr], 0.001)
        p = 1 / (1 + np.exp(-_margin(mdl, X[te])))
        leaked = len(set(g[te]) & set(g[tr]))
        official = {"metrics": metrics(p, y[te]), "test_sources": int(len(set(g[te]))),
                    "test_sources_also_in_train": int(leaked),
                    "note": "Dataset's own split. Inflated when test sources also appear in training."}
        log(f"[probe] official split: acc={official['metrics']['accuracy']:.3f} AUC={official['metrics']['roc_auc']}; "
            f"{leaked}/{official['test_sources']} test sources also in train")

    region = region_diagnostic(items, backbone, log=log) if any(it.get("regions") for it in items) else None

    # Final model on all data; C chosen by LOSO on all data; temperature from all OOF margins.
    C = _choose_C(X, y, g)
    mu, sd, clf = _fit(X, y, C)
    T = _temperature(ev["oof_margin"], y)
    probe = make_probe_module(backbone)
    with torch.no_grad():
        probe.mu.copy_(torch.tensor(mu, dtype=torch.float32))
        probe.sd.copy_(torch.tensor(sd, dtype=torch.float32))
        probe.head.weight.copy_(torch.tensor(clf.coef_, dtype=torch.float32))
        probe.head.bias.copy_(torch.tensor(clf.intercept_, dtype=torch.float32))
    torch.save(probe.state_dict(), out_dir / "model.pt")

    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(X)
    np.savez_compressed(out_dir / "ood_stats.npz", mean=lw.location_.astype(np.float32),
                        precision=lw.precision_.astype(np.float32))
    d = X - lw.location_
    dist = np.sqrt(np.einsum("ij,jk,ik->i", d, lw.precision_, d))
    # in-sample distances are optimistic; use LOSO distances for thresholds
    loso_d = np.zeros(len(y))
    for gg in np.unique(g):
        te = g == gg
        l2 = LedoitWolf().fit(X[~te])
        dd = X[te] - l2.location_
        loso_d[te] = np.sqrt(np.einsum("ij,jk,ik->i", dd, l2.precision_, dd))

    cfg = {
        "arch": "efficientnet_b0_tf_probe",
        "method": "linear_probe",
        "img_size": args.img_size,
        "classes": CLASSES,
        "positive_class": "forged",
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "temperature": T,
        "regularisation_C": C,
        "calibration": "temperature scaling fitted on leave-one-source-out margins (evaluated nested)",
        "ood": {"file": "ood_stats.npz", "val_p95": float(np.percentile(loso_d, 95)),
                "val_p99": float(np.percentile(loso_d, 99)),
                "note": "thresholds from leave-one-source-out distances"},
        "validation": g_res,
        "metrics": {"evaluation": "nested leave-one-source-out",
                    "test_uncalibrated": m_raw, "test_calibrated": m_cal,
                    "per_source_test": per_source(p_cal, y, g),
                    "official_split": official, "region_diagnostic": region,
                    "C_per_fold": ev["C_per_fold"]},
        "split_sizes": {"images": len(items), "sources": int(len(set(g)))},
        "dataset": {"path": str(args.data_dir), "report": "data_report.json", "name": args.dataset_name},
        "backbone_weights": EFFNET_URL,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "seed": args.seed,
    }
    (out_dir / "model_config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "evaluation.json").write_text(json.dumps({"validation": g_res, **cfg["metrics"]}, indent=2))
    (out_dir / "oof_predictions.json").write_text(json.dumps(
        [{"file": it["rel"], "source": it["source_group"], "label": it["label"], "p_forged_calibrated": round(float(p), 4)}
         for it, p in zip(items, p_cal)], indent=1))
    log(f"[done] artifacts written to {out_dir}")
    return cfg
