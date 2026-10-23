# Model artifacts

Output of `models/train.py` (or set `HALLMARK_MODEL_DIR` to another folder).

| File | In git? | Purpose |
|---|---|---|
| `model_config.json` | yes | architecture, calibration, OOD thresholds, **validation gate result**, all metrics |
| `evaluation.json`, `data_report.json`, `oof_predictions.json`, `reliability_test.png` | yes | evaluation records behind `docs/EVALUATION.md` |
| `model.pt` (~21 MB), `ood_stats.npz` (~6 MB) | no | weights; publish as a GitHub Release asset if needed |

The current records are from the Kaggle certificate-forgery dataset. That model **failed validation**, so the
app reports the AI check as *not used* using `model_config.json` alone; the weights are not needed for that.
Re-create everything with `python models/train.py --data-dir <dataset> --out models/artifacts`.
