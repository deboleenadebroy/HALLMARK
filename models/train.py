"""Train Hallmark's visual forgery classifier.

Designed for the Kaggle "Certificate Forgery Dataset"
(https://www.kaggle.com/datasets/harikrish03/certificate-forgery-dataset) but
works with any folder of labelled certificate images.

Two training methods (``--method auto`` picks one):

* ``linear_probe`` (models/linear_probe.py) for datasets with few independent
  source documents: frozen ImageNet EfficientNet-B0 + regularised logistic
  regression, evaluated with nested leave-one-source-out cross-validation.
  This is what the Kaggle dataset needs (254 images, 15 source documents).
* ``finetune`` for larger datasets, described below.

Supported layouts: class folders (genuine/, fake/ ...), a CSV of labels, or a
YOLO / Roboflow export (data.yaml + images/ + labels/), where an image is
"forged" if any labelled region belongs to a forged class.

Fine-tuning pipeline
  1. Inspect the data: classes, counts, formats, sizes, exact and near duplicates.
  2. Group near-duplicates (perceptual hash) so the same template / source image
     never lands in more than one split (reduces template leakage).
  3. Grouped, stratified train / validation / test split (70 / 15 / 15).
  4. Transfer learning (EfficientNet-B0 by default); augmentation on train only.
  5. Temperature-scaling calibration on the validation split.
  6. Evaluation on the untouched test split: accuracy, precision, recall, F1,
     confusion matrix, ROC-AUC, Brier score, ECE — before and after calibration.
  7. Out-of-distribution statistics (Mahalanobis distance in feature space)
     so inference can flag certificates unlike anything seen in training.

Usage (e.g. in a Kaggle notebook with the dataset attached):

    python models/train.py --data-dir /kaggle/input/certificate-forgery-dataset \
                           --out models/artifacts --epochs 15

Inspection only (no training):

    python models/train.py --data-dir <dir> --inspect-only

Nothing here runs during certificate analysis; inference lives in predict.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
AUTH_WORDS = re.compile(r"(?i)^(genuine|real|original|authentic|valid|true|legit\w*|unaltered|pristine|0)$")
FORGED_WORDS = re.compile(r"(?i)^(fake|forged?|forgery|forgeries|tampered|manipulated|altered|counterfeit|"
                          r"fraud\w*|invalid|false|edited|spliced|1)$")
CLASSES = ["authentic", "forged"]          # index 1 = positive class (forged)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- discovery & inspection

def _label_from_path(rel: Path, label_map: dict[str, str]) -> str | None:
    for part in reversed(rel.parts[:-1]):
        key = part.lower()
        if key in label_map:
            return label_map[key]
        if AUTH_WORDS.match(part):
            return "authentic"
        if FORGED_WORDS.match(part):
            return "forged"
    # fall back to filename tokens, e.g. "fake_0012.png"
    tokens = re.split(r"[_\-. ]+", rel.stem)
    for t in tokens:
        if t.lower() in label_map:
            return label_map[t.lower()]
        if AUTH_WORDS.match(t) and not t.isdigit():
            return "authentic"
        if FORGED_WORDS.match(t) and not t.isdigit():
            return "forged"
    return None


def discover(data_dir: Path, label_map: dict[str, str], labels_csv: Path | None = None):
    items, unlabelled = [], []
    csv_labels = {}
    if labels_csv:
        import csv
        with open(labels_csv, newline="") as f:
            for row in csv.DictReader(f):
                fn = row.get("filename") or row.get("file") or row.get("image") or row.get("path")
                lab = (row.get("label") or row.get("class") or "").strip().lower()
                lab = label_map.get(lab, lab)
                if AUTH_WORDS.match(lab):
                    lab = "authentic"
                elif FORGED_WORDS.match(lab):
                    lab = "forged"
                if fn:
                    csv_labels[Path(fn).name] = lab
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        rel = p.relative_to(data_dir)
        lab = csv_labels.get(p.name) if labels_csv else _label_from_path(rel, label_map)
        if lab in CLASSES:
            items.append({"path": str(p), "rel": str(rel), "label": lab,
                          "source": rel.parts[0] if len(rel.parts) > 2 else "root"})
        else:
            unlabelled.append(str(rel))
    return items, unlabelled


ROBOFLOW_STEM = re.compile(r"^(?P<src>.+?)(?:-Copy-\d+-)?_(?:jpg|jpeg|png|bmp|webp)\.rf\.[0-9a-f]{16,}$", re.I)


def roboflow_source(path: str) -> str | None:
    """Roboflow exports name files '<original>_<ext>.rf.<hash>'; recover the original document."""
    m = ROBOFLOW_STEM.match(Path(path).stem)
    return m.group("src") if m else None


def find_yolo_yaml(data_dir: Path) -> Path | None:
    for p in sorted(data_dir.rglob("*.yaml")) + sorted(data_dir.rglob("*.yml")):
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        if re.search(r"(?m)^names\s*:", txt):
            return p
    return None


def discover_yolo(yaml_path: Path, label_map: dict[str, str]):
    import yaml
    cfg = yaml.safe_load(yaml_path.read_text())
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    forged_ids = set()
    for i, n in enumerate(names):
        lab = label_map.get(str(n).lower())
        if lab == "forged" or (lab is None and FORGED_WORDS.match(str(n))):
            forged_ids.add(i)
    if not forged_ids:
        raise SystemExit(f"No forged class among YOLO names {names}; pass --label-map 'name=forged'.")
    root = yaml_path.parent
    items, unlabelled = [], []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMG_EXT or "images" not in img.parts:
            continue
        parts = list(img.parts)
        k = len(parts) - 1 - parts[::-1].index("images")
        lab_path = Path(*parts[:k], "labels", *parts[k + 1:]).with_suffix(".txt")
        if not lab_path.exists():
            unlabelled.append(str(img.relative_to(root)))
            continue
        regions = []
        for line in lab_path.read_text().splitlines():
            f = line.split()
            if len(f) >= 5:
                c = int(float(f[0]))
                regions.append({"class": names[c] if c < len(names) else str(c), "forged": c in forged_ids,
                                "cx": float(f[1]), "cy": float(f[2]), "w": float(f[3]), "h": float(f[4])})
        rel = img.relative_to(root)
        split = parts[k - 1] if k >= 1 else ""
        items.append({"path": str(img), "rel": str(rel),
                      "label": "forged" if any(r["forged"] for r in regions) else "authentic",
                      "source": split, "official_split": {"valid": "val"}.get(split, split),
                      "regions": regions})
    info = {"format": "yolo", "yaml": str(yaml_path.name), "class_names": names,
            "forged_class_ids": sorted(forged_ids),
            "image_label_rule": "forged if any labelled region is of a forged class",
            "license": (cfg.get("roboflow") or {}).get("license"),
            "source_url": (cfg.get("roboflow") or {}).get("url")}
    return items, unlabelled, info


def dhash(img: Image.Image, size: int = 8) -> np.ndarray:
    """Difference hash as a packed uint8 array (size*size bits)."""
    g = ImageOps.grayscale(img).resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def _near_dup_groups(hashes: list, n: int, threshold: int, seed_pairs: list[tuple[int, int]]):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in seed_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    ids = [i for i, h in enumerate(hashes) if h is not None]
    if ids:
        H = np.stack([hashes[i] for i in ids])
        pairs = 0
        for a in range(len(ids)):
            d = np.unpackbits(np.bitwise_xor(H[a], H[a + 1:]), axis=1).sum(axis=1)
            for off in np.nonzero(d <= threshold)[0]:
                ra, rb = find(ids[a]), find(ids[a + 1 + off])
                pairs += 1
                if ra != rb:
                    parent[rb] = ra
    else:
        pairs = 0
    return [find(i) for i in range(n)], pairs


def _mutual_nn_pairs(hashes: list, max_dist: int = 12) -> list[tuple[int, int]]:
    """Pair images that are each other's nearest neighbour (likely genuine/forged derivations)."""
    ids = [i for i, h in enumerate(hashes) if h is not None]
    if len(ids) < 2:
        return []
    H = np.stack([hashes[i] for i in ids])
    D = np.stack([np.unpackbits(np.bitwise_xor(H[a], H), axis=1).sum(axis=1) for a in range(len(ids))])
    np.fill_diagonal(D, 10 ** 6)
    nn = D.argmin(axis=1)
    return [(ids[a], ids[b]) for a, b in enumerate(nn) if nn[b] == a and a < b and D[a, b] <= max_dist]


def inspect(items: list[dict]) -> tuple[dict, list[int], list[list[int]]]:
    fmt, sizes, sha_to_idx = Counter(), [], defaultdict(list)
    h8, h16, bad = [], [], []
    for i, it in enumerate(items):
        try:
            raw = Path(it["path"]).read_bytes()
            sha_to_idx[hashlib.sha256(raw).hexdigest()].append(i)
            with Image.open(it["path"]) as im:
                fmt[im.format] += 1
                sizes.append(im.size)
                rgb = im.convert("RGB")
                h8.append(dhash(rgb, 8))
                h16.append(dhash(rgb, 16))
        except Exception as exc:
            bad.append({"file": it["rel"], "error": str(exc)[:120]})
            h8.append(None)
            h16.append(None)
    exact_pairs = [(v[0], j) for v in sha_to_idx.values() for j in v[1:]]
    n = len(items)
    # Coarse 64-bit hash first; if one group swallows the dataset (shared template),
    # retry with a finer 256-bit hash; if that also fails, fall back to exact duplicates.
    # Cascade from coarse to fine: the first grouping in which no single group exceeds 20%
    # of the data is used. Fine thresholds still keep a forgery together with the genuine
    # image it was derived from (they differ only in a small region).
    leakage_controlled = True
    for strategy, hs, thr in (("dhash64<=6", h8, 6), ("dhash256<=12", h16, 12),
                              ("dhash256<=4", h16, 4), ("mutual-nearest-neighbour-pairs", None, 0),
                              ("exact-duplicates-only", [None] * n, 0)):
        if hs is None:
            groups, near_pairs = _near_dup_groups([None] * n, n, 0, exact_pairs + _mutual_nn_pairs(h16))
        else:
            groups, near_pairs = _near_dup_groups(hs, n, thr, exact_pairs)
        largest = max(Counter(groups).values()) if n else 0
        if not n or largest <= 0.2 * n:
            break
    if strategy in ("exact-duplicates-only", "mutual-nearest-neighbour-pairs"):
        leakage_controlled = False   # only partial control: shared templates still cross splits
    conflicting = [v for v in sha_to_idx.values() if len({items[i]["label"] for i in v}) > 1]
    w = [s[0] for s in sizes] or [0]
    h = [s[1] for s in sizes] or [0]
    report = {
        "n_images": n,
        "class_counts": dict(Counter(it["label"] for it in items)),
        "source_counts": dict(Counter(it["source"] for it in items)),
        "formats": dict(fmt),
        "width": {"min": int(min(w)), "median": int(np.median(w)), "max": int(max(w))},
        "height": {"min": int(min(h)), "median": int(np.median(h)), "max": int(max(h))},
        "exact_duplicate_files": len(exact_pairs),
        "exact_duplicates_with_conflicting_labels": len(conflicting),
        "near_duplicate_pairs": near_pairs,
        "grouping_strategy": strategy,
        "template_leakage_controlled": leakage_controlled,
        "duplicate_groups": len(set(groups)),
        "largest_group": int(max(Counter(groups).values())) if n else 0,
        "unreadable_files": bad,
    }
    if not leakage_controlled:
        print(f"[data] WARNING: images share templates too closely for full near-duplicate grouping; "
              f"using '{strategy}'. Test metrics may be optimistic about unseen layouts.")
    return report, groups, conflicting


def grouped_split(items, groups, seed: int, fractions=(0.70, 0.15, 0.15)):
    """Assign whole groups to splits, stratified by each group's majority label."""
    rnd = random.Random(seed)
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    by_label = defaultdict(list)
    for g, idxs in by_group.items():
        maj = Counter(items[i]["label"] for i in idxs).most_common(1)[0][0]
        by_label[maj].append(g)
    split = {}
    for lab, gs in by_label.items():
        rnd.shuffle(gs)
        total = sum(len(by_group[g]) for g in gs)
        acc = 0
        for g in gs:
            frac = acc / max(1, total)
            name = "train" if frac < fractions[0] else ("val" if frac < fractions[0] + fractions[1] else "test")
            for i in by_group[g]:
                split[i] = name
            acc += len(by_group[g])
    return split


# --------------------------------------------------------------------------- model / data (torch)

def build_model(arch: str, pretrained: bool):
    import torch.nn as nn
    import torchvision.models as tvm

    if arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif arch == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    elif arch == "efficientnet_b0_tf_probe":
        from models.linear_probe import make_probe_module
        return make_probe_module()
    elif arch == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    else:
        raise ValueError(f"unknown architecture {arch}")
    return m


def features_and_logits(model, x):
    """Return (penultimate features, logits) for supported models."""
    import torch
    if getattr(model, "is_linear_probe", False):
        f = model.features(x)
        z = model.head((f - model.mu) / model.sd)
        return f, torch.cat([torch.zeros_like(z), z], 1)
    f = model.features(x)
    f = model.avgpool(f)
    f = torch.flatten(f, 1)
    if hasattr(model.classifier, "__len__") and len(model.classifier) == 4:   # mobilenet_v3
        h = model.classifier[0](f)
        h = model.classifier[1](h)
        feats = h
        logits = model.classifier[3](model.classifier[2](h))
    else:                                                                      # efficientnet
        feats = f
        logits = model.classifier(f)
    return feats, logits


def letterbox(img: Image.Image, size: int, fill=(255, 255, 255)) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = size / max(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def make_transforms(img_size: int, train: bool):
    import torch
    import torchvision.transforms.v2 as T

    base = [T.Lambda(lambda im: letterbox(im, img_size))]
    if train:
        base += [
            T.RandomApply([T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05),
                                          fill=255)], p=0.7),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.15, 0.02)], p=0.7),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.2),
            T.PILToTensor(),
            T.RandomApply([T.JPEG((50, 95))], p=0.4),
        ]
    else:
        base += [T.PILToTensor()]
    base += [T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(base)


class ImageDataset:
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        with Image.open(it["path"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        return self.transform(im), CLASSES.index(it["label"])


# --------------------------------------------------------------------------- metrics

def ece_score(probs_pos: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = np.where(probs_pos >= 0.5, probs_pos, 1 - probs_pos)
    pred = (probs_pos >= 0.5).astype(int)
    edges = np.linspace(0.5, 1.0, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return float(e)


def metrics(probs_pos: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                                 log_loss, precision_score, recall_score, roc_auc_score)
    pred = (probs_pos >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_forged": float(precision_score(y, pred, zero_division=0)),
        "recall_forged": float(recall_score(y, pred, zero_division=0)),
        "f1_forged": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": "rows=true [authentic, forged], cols=predicted [authentic, forged]",
        "brier": float(brier_score_loss(y, probs_pos)),
        "log_loss": float(log_loss(y, np.clip(probs_pos, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "ece_15": ece_score(probs_pos, y),
    }
    out["roc_auc"] = float(roc_auc_score(y, probs_pos)) if len(set(y.tolist())) == 2 else None
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    import torch
    lg = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss
    opt.step(closure)
    t = float(log_t.exp().item())
    return float(min(max(t, 0.05), 20.0))


def softmax_pos(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def _plots(out_dir: Path, probs, y, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    edges = np.linspace(0, 1, 11)
    mids, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if sel.any():
            mids.append(probs[sel].mean())
            accs.append(y[sel].mean())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="#999")
    ax.plot(mids, accs, "o-", color="#6D2932")
    ax.set_xlabel("Predicted P(forged)")
    ax.set_ylabel("Observed fraction forged")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "reliability_test.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- main

def run(args) -> dict:
    import torch
    from torch.utils.data import DataLoader

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_map = {}
    for kv in (args.label_map or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            label_map[k.strip().lower()] = v.strip().lower()

    yolo_yaml = None if args.labels_csv else find_yolo_yaml(data_dir)
    fmt_info = {"format": "csv" if args.labels_csv else "folders"}
    if yolo_yaml:
        items, unlabelled, fmt_info = discover_yolo(yolo_yaml, label_map)
        print(f"[data] YOLO dataset ({yolo_yaml.name}): classes {fmt_info['class_names']}, "
              f"forged ids {fmt_info['forged_class_ids']}")
    else:
        items, unlabelled = discover(data_dir, label_map, Path(args.labels_csv) if args.labels_csv else None)
    print(f"[data] {len(items)} labelled images, {len(unlabelled)} unlabelled")
    if unlabelled:
        print("[data] example unlabelled paths:", unlabelled[:10])
    counts = Counter(it["label"] for it in items)
    if len(counts) < 2:
        folders = sorted({str(Path(u).parent) for u in unlabelled})[:20]
        raise SystemExit("Could not find both classes. Folders seen: "
                         f"{folders}. Pass --label-map 'folderA=authentic,folderB=forged' or --labels-csv.")

    report, groups, conflicting = inspect(items)
    report["unlabelled_files"] = len(unlabelled)
    if conflicting:
        drop = {i for v in conflicting for i in v}
        print(f"[data] dropping {len(drop)} files that are exact duplicates with conflicting labels")
        keep = [i for i in range(len(items)) if i not in drop]
        items = [items[i] for i in keep]
        groups = [groups[i] for i in keep]
    bad = {b["file"] for b in report["unreadable_files"]}
    if bad:
        keep = [i for i, it in enumerate(items) if it["rel"] not in bad]
        items, groups = [items[i] for i in keep], [groups[i] for i in keep]
    # Independent source documents: Roboflow file names if present, else perceptual-hash groups.
    rf = [roboflow_source(it["path"]) for it in items]
    if items and all(rf):
        for it, src in zip(items, rf):
            it["source_group"] = src
        report["source_grouping"] = "roboflow original file name"
    else:
        for it, gid in zip(items, groups):
            it["source_group"] = f"group_{gid}"
        report["source_grouping"] = report["grouping_strategy"]
    src_label = {}
    for it in items:
        src_label.setdefault(it["source_group"], Counter())[it["label"]] += 1
    report["n_sources"] = len(src_label)
    report["sources"] = {k: {"images": sum(v.values()), "labels": dict(v)} for k, v in sorted(src_label.items())}
    report["sources_per_class"] = dict(Counter(max(v, key=v.get) for v in src_label.values()))
    report["format"] = fmt_info
    if fmt_info.get("format") == "yolo":
        report["region_counts"] = dict(Counter(r["class"] for it in items for r in it["regions"]))
        report["regions_per_image"] = dict(Counter(len(it["regions"]) for it in items))
    off = {}
    for it in items:
        if it.get("official_split"):
            off.setdefault(it["official_split"], set()).add(it["source_group"])
    if "test" in off:
        train_src = set().union(*(v for k, v in off.items() if k != "test"))
        report["official_split_leakage"] = {
            "split_sizes": {k: sum(1 for it in items if it.get("official_split") == k) for k in off},
            "test_sources": len(off["test"]), "test_sources_also_in_train_or_val": len(off["test"] & train_src)}
    (out_dir / "data_report.json").write_text(json.dumps(report, indent=2))
    print("[data] report:", json.dumps({k: report.get(k) for k in
                                        ("class_counts", "n_sources", "sources_per_class", "formats",
                                         "exact_duplicate_files", "source_grouping", "official_split_leakage")}))
    if args.inspect_only:
        return {"data_report": report}

    method = args.method
    if method == "auto":
        method = "linear_probe" if report["n_sources"] < 50 else "finetune"
        print(f"[train] method auto → {method} ({report['n_sources']} independent sources)")
    if method == "linear_probe":
        from models.linear_probe import run_probe
        return run_probe(items, report, args, out_dir)
    groups = [it["source_group"] for it in items]

    split = grouped_split(items, groups, args.seed)
    parts = {s: [items[i] for i in range(len(items)) if split[i] == s] for s in ("train", "val", "test")}
    for s, v in parts.items():
        print(f"[split] {s}: {len(v)} {dict(Counter(it['label'] for it in v))}")
    if min(len(v) for v in parts.values()) == 0:
        raise SystemExit("A split is empty; the dataset is too small for a grouped split.")
    (out_dir / "splits.json").write_text(json.dumps({s: [it["rel"] for it in v] for s, v in parts.items()}, indent=1))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("[train] device:", device)
    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)

    nw = 0 if args.workers == 0 else args.workers
    dl = lambda s, tr: DataLoader(ImageDataset(parts[s], make_transforms(args.img_size, tr)),
                                  batch_size=args.batch_size, shuffle=tr, num_workers=nw,
                                  pin_memory=device.type == "cuda")
    train_dl, val_dl, test_dl = dl("train", True), dl("val", False), dl("test", False)
    train_eval_dl = dl("train", False)

    c = Counter(it["label"] for it in parts["train"])
    w = torch.tensor([len(parts["train"]) / (2 * max(1, c[k])) for k in CLASSES], dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, args.epochs * len(train_dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def collect(loader):
        model.eval()
        L, F, Y = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    f, lg = features_and_logits(model, x)
                L.append(lg.float().cpu().numpy())
                F.append(f.float().cpu().numpy())
                Y.append(y.numpy())
        return np.concatenate(L), np.concatenate(F), np.concatenate(Y)

    best, best_state, patience, history = math.inf, None, 0, []
    for ep in range(args.epochs):
        model.train()
        t0, tl, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y)
            n += len(y)
        vl, _, vy = collect(val_dl)
        vloss = float(torch.nn.functional.cross_entropy(torch.tensor(vl), torch.tensor(vy)).item())
        vacc = float(((vl.argmax(1)) == vy).mean())
        history.append({"epoch": ep + 1, "train_loss": tl / max(1, n), "val_loss": vloss, "val_acc": vacc,
                        "seconds": round(time.time() - t0, 1)})
        print(f"[train] epoch {ep + 1}/{args.epochs} train_loss={tl / max(1, n):.4f} "
              f"val_loss={vloss:.4f} val_acc={vacc:.4f}")
        if vloss < best - 1e-4:
            best, patience = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print("[train] early stopping")
                break
    model.load_state_dict(best_state)

    # calibration on validation, evaluation on test
    vl, vf, vy = collect(val_dl)
    T = fit_temperature(vl, vy)
    tl_, tf, ty = collect(test_dl)
    trl, trf, _ = collect(train_eval_dl)
    test_raw = metrics(softmax_pos(tl_, 1.0), ty)
    test_cal = metrics(softmax_pos(tl_, T), ty)
    from sklearn.metrics import balanced_accuracy_score
    from models.linear_probe import gate
    for mm, pp in ((test_raw, softmax_pos(tl_, 1.0)), (test_cal, softmax_pos(tl_, T))):
        mm["balanced_accuracy"] = float(balanced_accuracy_score(ty, pp >= 0.5))
    validation = gate(test_cal, ty, np.array([it["source_group"] for it in parts["test"]]))
    print(f"[eval] validation gate: {'PASSED' if validation['passed'] else 'FAILED'} {validation['reasons']}")
    val_cal = metrics(softmax_pos(vl, T), vy)
    _plots(out_dir, softmax_pos(tl_, T), ty, "Test reliability (calibrated)")

    # per-source generalisation (if the dataset has top-level sub-collections)
    per_source = {}
    srcs = [it["source"] for it in parts["test"]]
    for s in sorted(set(srcs)):
        sel = np.array([x == s for x in srcs])
        if sel.sum() >= 10:
            per_source[s] = metrics(softmax_pos(tl_[sel], T), ty[sel])

    # OOD statistics: Mahalanobis distance with Ledoit-Wolf covariance on train features
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(trf)
    prec = lw.precision_.astype(np.float32)
    mu = lw.location_.astype(np.float32)

    def maha(F):
        d = F - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))
    vd = maha(vf)
    np.savez_compressed(out_dir / "ood_stats.npz", mean=mu, precision=prec)

    torch.save(model.state_dict(), out_dir / "model.pt")
    cfg = {
        "arch": args.arch,
        "img_size": args.img_size,
        "classes": CLASSES,
        "positive_class": "forged",
        "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "temperature": T,
        "method": "finetune",
        "calibration": "temperature scaling fitted on the validation split",
        "validation": validation,
        "ood": {"file": "ood_stats.npz", "val_p95": float(np.percentile(vd, 95)),
                "val_p99": float(np.percentile(vd, 99))},
        "metrics": {"test_uncalibrated": test_raw, "test_calibrated": test_cal, "val_calibrated": val_cal,
                    "per_source_test": per_source},
        "split_sizes": {s: len(v) for s, v in parts.items()},
        "history": history,
        "dataset": {"path": str(data_dir), "report": "data_report.json",
                    "name": args.dataset_name},
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "seed": args.seed,
    }
    (out_dir / "model_config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "evaluation.json").write_text(json.dumps(cfg["metrics"], indent=2))
    print("[eval] test (calibrated):", json.dumps({k: test_cal[k] for k in
                                                  ("accuracy", "precision_forged", "recall_forged",
                                                   "f1_forged", "roc_auc", "brier", "ece_15")}))
    print(f"[done] artifacts written to {out_dir}")
    return cfg


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "artifacts"))
    ap.add_argument("--method", default="auto", choices=["auto", "linear_probe", "finetune"],
                    help="auto: linear_probe when fewer than 50 independent source documents")
    ap.add_argument("--pretrained-weights", default=None,
                    help="local path to efficientnet-b0-355c32eb.pth (linear probe); downloaded from GitHub if omitted")
    ap.add_argument("--arch", default="efficientnet_b0",
                    choices=["efficientnet_b0", "mobilenet_v3_large", "mobilenet_v3_small"])
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-map", default="", help="e.g. 'genuine_certs=authentic,fake_certs=forged'")
    ap.add_argument("--labels-csv", default=None, help="CSV with filename,label columns")
    ap.add_argument("--dataset-name", default="Kaggle: harikrish03/certificate-forgery-dataset")
    ap.add_argument("--no-pretrained", action="store_true", help="do not download ImageNet weights")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--inspect-only", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    run(parse_args())
