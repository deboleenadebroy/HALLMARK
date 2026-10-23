import json
import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from models import predict
from services.document_reader import load_document
from services.evidence import Status
from tests.fixtures import certificate_image, image_bytes

warnings.filterwarnings("ignore")

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


class MissingModel(unittest.TestCase):
    def test_missing_model_not_performed(self):
        d = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {"HALLMARK_MODEL_DIR": d}):
            predict.reset_cache()
            doc = load_document("c.png", image_bytes(certificate_image()))
            ev = predict.analyse_visual(doc.pages)
        predict.reset_cache()
        self.assertEqual(ev.status, Status.NOT_PERFORMED)
        self.assertIn("No trained visual model", ev.explanation)
        self.assertEqual(ev.findings, [])


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TrainAndPredictSmoke(unittest.TestCase):
    """Runs the real training code on a tiny synthetic dataset. Proves the code
    path works; the resulting metrics say nothing about real-world accuracy."""

    @classmethod
    def setUpClass(cls):
        from models.train import parse_args, run
        from tests.make_synthetic_dataset import build
        cls.tmp = Path(tempfile.mkdtemp(prefix="hallmark-train-"))
        build(cls.tmp / "data", n_per_class=16, size=(480, 336))
        cls.out = cls.tmp / "model"
        cls.cfg = run(parse_args(["--data-dir", str(cls.tmp / "data"), "--out", str(cls.out),
                                  "--method", "finetune",
                                  "--arch", "mobilenet_v3_small", "--img-size", "128", "--epochs", "1",
                                  "--batch-size", "8", "--workers", "0", "--no-pretrained", "--cpu",
                                  "--dataset-name", "synthetic smoke test"]))

    @classmethod
    def tearDownClass(cls):
        predict.reset_cache()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_artifacts_and_metrics(self):
        for f in ("model.pt", "model_config.json", "ood_stats.npz", "data_report.json", "splits.json"):
            self.assertTrue((self.out / f).exists(), f)
        m = json.loads((self.out / "model_config.json").read_text())["metrics"]["test_calibrated"]
        for k in ("accuracy", "precision_forged", "recall_forged", "f1_forged", "confusion_matrix", "brier", "ece_15"):
            self.assertIn(k, m)
        self.assertGreater(self.cfg["temperature"], 0)

    def test_splits_disjoint(self):
        sp = json.loads((self.out / "splits.json").read_text())
        self.assertFalse(set(sp["train"]) & set(sp["test"]))
        self.assertFalse(set(sp["train"]) & set(sp["val"]))

    def _evidence(self, override=True):
        env = {"HALLMARK_MODEL_DIR": str(self.out)}
        if override:
            env["HALLMARK_ALLOW_UNVALIDATED_MODEL"] = "1"
        with mock.patch.dict(os.environ, env):
            predict.reset_cache()
            doc = load_document("c.png", image_bytes(certificate_image()))
            return predict.analyse_visual(doc.pages)

    def test_prediction_evidence(self):
        ev = self._evidence()
        self.assertIn(ev.status, (Status.PASS, Status.FAIL, Status.INCONCLUSIVE))
        p = ev.details["pages"][0]
        self.assertAlmostEqual(p["p_forged"] + p["p_authentic"], 1.0, places=5)
        self.assertIn(p["ood"], ("in_distribution", "borderline", "out_of_distribution"))

    def test_failed_validation_gate_not_used(self):
        cfg = json.loads((self.out / "model_config.json").read_text())
        self.assertFalse(cfg["validation"]["passed"])          # 16 synthetic sources/class, 1 epoch
        os.environ.pop("HALLMARK_ALLOW_UNVALIDATED_MODEL", None)
        ev = self._evidence(override=False)
        self.assertEqual(ev.status, Status.NOT_PERFORMED)
        self.assertIn("failed held-out validation", ev.explanation)
        ev2 = self._evidence(override=True)
        self.assertIn(ev2.status, (Status.PASS, Status.FAIL, Status.INCONCLUSIVE))
        self.assertTrue(ev2.limitations[0].startswith("RESEARCH OVERRIDE"))

    def test_out_of_distribution_flag(self):
        cfg_path = self.out / "model_config.json"
        cfg = json.loads(cfg_path.read_text())
        orig = cfg_path.read_text()
        cfg["ood"]["val_p95"] = cfg["ood"]["val_p99"] = 0.0
        cfg_path.write_text(json.dumps(cfg))
        try:
            ev = self._evidence()
        finally:
            cfg_path.write_text(orig)
            predict.reset_cache()
        self.assertEqual(ev.status, Status.INCONCLUSIVE)
        self.assertEqual(ev.details["pages"][0]["ood"], "out_of_distribution")


def _yolo_dataset(root: Path, n_src: int = 4, copies: int = 3) -> Path:
    """Tiny Roboflow-style YOLO export: per source document several '<src>_jpg.rf.<hash>' copies."""
    import hashlib
    from tests.fixtures import certificate_image
    (root / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
                                    "nc: 2\nnames: ['fake', 'true']\n")
    splits = ["train", "valid", "test"]
    for s in splits:
        (root / s / "images").mkdir(parents=True)
        (root / s / "labels").mkdir(parents=True)
    k = 0
    for src in range(n_src):
        for label in ("g", "f"):
            for c in range(copies):
                split = splits[k % 3]
                k += 1
                h = hashlib.md5(f"{src}{label}{c}".encode()).hexdigest()
                stem = f"{label}doc{src}_jpg.rf.{h}"
                img = certificate_image(seed=src).resize((128, 90))
                if label == "f":
                    img.paste((200, 40, 40), (10, 10, 40, 30))
                img.save(root / split / "images" / f"{stem}.jpg")
                cls = 0 if label == "f" else 1
                (root / split / "labels" / f"{stem}.txt").write_text(
                    f"{cls} 0.2 0.2 0.2 0.2\n1 0.8 0.8 0.2 0.2\n1 0.5 0.9 0.2 0.1\n")
    return root


class YoloDiscovery(unittest.TestCase):
    def test_yolo_labels_and_sources(self):
        from models.train import discover_yolo, find_yolo_yaml, roboflow_source
        root = _yolo_dataset(Path(tempfile.mkdtemp()))
        items, unl, info = discover_yolo(find_yolo_yaml(root), {})
        self.assertEqual(len(items), 4 * 2 * 3)
        self.assertEqual(info["forged_class_ids"], [0])
        self.assertEqual(sum(it["label"] == "forged" for it in items), 12)
        self.assertEqual(len({roboflow_source(it["path"]) for it in items}), 8)
        self.assertEqual(roboflow_source("Upto-Markscard_page-0002-Copy-2-_jpg.rf.2e1e63d1b985e32e0e76a8f8fe38afea.jpg"),
                         "Upto-Markscard_page-0002")
        self.assertTrue(all(len(it["regions"]) == 3 for it in items))
        shutil.rmtree(root, ignore_errors=True)

    def test_inspect_reports_official_split_leakage(self):
        from models.train import parse_args, run
        root = _yolo_dataset(Path(tempfile.mkdtemp()))
        out = root / "out"
        run(parse_args(["--data-dir", str(root), "--out", str(out), "--inspect-only"]))
        rep = json.loads((out / "data_report.json").read_text())
        self.assertEqual(rep["n_sources"], 8)
        self.assertEqual(rep["source_grouping"], "roboflow original file name")
        self.assertEqual(rep["official_split_leakage"]["test_sources_also_in_train_or_val"],
                         rep["official_split_leakage"]["test_sources"])
        shutil.rmtree(root, ignore_errors=True)


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class LinearProbeSmoke(unittest.TestCase):
    def test_probe_trains_evaluates_and_predicts(self):
        try:
            import efficientnet_pytorch  # noqa: F401
        except Exception:
            self.skipTest("efficientnet_pytorch not installed")
        from models.train import parse_args, run
        root = _yolo_dataset(Path(tempfile.mkdtemp()))
        out = root / "model"
        cfg = run(parse_args(["--data-dir", str(root), "--out", str(out), "--method", "linear_probe",
                              "--img-size", "96", "--no-pretrained"]))
        self.assertEqual(cfg["metrics"]["evaluation"], "nested leave-one-source-out")
        self.assertIn("validation", cfg)
        self.assertEqual(len(cfg["metrics"]["per_source_test"]), 8)
        self.assertIsNotNone(cfg["metrics"]["official_split"])
        with mock.patch.dict(os.environ, {"HALLMARK_MODEL_DIR": str(out), "HALLMARK_ALLOW_UNVALIDATED_MODEL": "1"}):
            predict.reset_cache()
            doc = load_document("c.png", image_bytes(certificate_image()))
            ev = predict.analyse_visual(doc.pages)
        predict.reset_cache()
        p = ev.details["pages"][0]
        self.assertAlmostEqual(p["p_forged"] + p["p_authentic"], 1.0, places=5)
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
