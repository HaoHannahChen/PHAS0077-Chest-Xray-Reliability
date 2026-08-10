"""Run a lightweight reproducibility check without the chest X-ray dataset."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (
    predictive_entropy,
    prediction_confidence,
    probability_margin,
    multiclass_brier_score,
    negative_log_likelihood,
)

sample = pd.read_csv(ROOT / "sample_data" / "sample_probabilities.csv")
expected = json.loads((ROOT / "results" / "expected_quick_check.json").read_text())

y_true = sample["true_label"].to_numpy(int)
probs = sample[["p0", "p1", "p2", "p3"]].to_numpy(float)
pred = probs.argmax(axis=1)

actual = {
    "accuracy": float((pred == y_true).mean()),
    "mean_confidence": float(prediction_confidence(probs).mean()),
    "mean_entropy": float(predictive_entropy(probs).mean()),
    "mean_margin": float(probability_margin(probs).mean()),
    "brier_score": multiclass_brier_score(y_true, probs),
    "nll": negative_log_likelihood(y_true, probs),
}

for key, expected_value in expected.items():
    if not np.isclose(actual[key], expected_value, atol=1e-10):
        raise AssertionError(f"{key}: expected {expected_value}, got {actual[key]}")

print("Quick reproducibility check passed.")
for key, value in actual.items():
    print(f"{key}: {value:.6f}")
