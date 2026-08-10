from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import math
import numpy as np
import torch

from src.metrics import predictive_entropy, probability_margin
from src.occlusion import mask_grid_cell


def test_uniform_four_class_entropy():
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(predictive_entropy(probs), math.log(4), atol=1e-10)


def test_near_deterministic_entropy_is_near_zero():
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    assert predictive_entropy(probs) < 1e-9


def test_probability_margin():
    probs = np.array([0.60, 0.25, 0.10, 0.05])
    assert np.isclose(probability_margin(probs), 0.35)


def test_7x7_mask_on_224_image():
    image = torch.ones((1, 1, 224, 224))
    masked = mask_grid_cell(image, 0, 0, grid_size=7, fill_value=0.0)
    assert torch.all(masked[:, :, :32, :32] == 0)
    assert torch.all(masked[:, :, 32:, :] == 1)
