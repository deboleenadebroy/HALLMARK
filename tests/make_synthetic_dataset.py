"""Build a tiny synthetic certificate dataset for smoke-testing train.py.

This is NOT a substitute for the real dataset: images are generated here and
'forgeries' are simple pasted patches. It only proves the training code runs.
"""
from __future__ import annotations

import io
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.fixtures import CERT_LINES, _font, certificate_image  # noqa: E402

NAMES = ["Priya Raman", "Arjun Mehta", "Sara Khan", "Li Wei", "Maria Lopez", "Tom Baker", "Anita Rao",
         "Kofi Mensah", "Rohan Das", "Emma Stone", "Ivan Petrov", "Aisha Bello"]
COURSES = ["Applied Machine Learning", "Data Structures", "Cloud Basics", "Financial Modelling",
           "Public Speaking", "Python Programming"]


def _noise(img, s, seed):
    a = np.asarray(img).astype(np.float32) + np.random.default_rng(seed).normal(0, s, (img.height, img.width, 3))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def build(out: Path, n_per_class: int = 24, size=(800, 560)) -> Path:
    rnd = random.Random(0)
    for cls in ("genuine", "forged"):
        (out / cls).mkdir(parents=True, exist_ok=True)
    for i in range(n_per_class):
        lines = list(CERT_LINES)
        lines[3] = (rnd.choice(NAMES), 52, True)
        lines[5] = (rnd.choice(COURSES), 38, True)
        img = certificate_image(lines, seed=i).resize(size)
        img = _noise(img, 3, i)
        img.save(out / "genuine" / f"g_{i:03d}.jpg", quality=88)
        f = img.copy()
        patch = Image.new("RGB", (260, 40), (255, 255, 255))
        ImageDraw.Draw(patch).text((5, 2), rnd.choice(NAMES), fill=(0, 0, 0), font=_font(26, True))
        f.paste(patch, (rnd.randint(250, 300), rnd.randint(180, 220)))
        f.save(out / "forged" / f"f_{i:03d}.jpg", quality=88)
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
