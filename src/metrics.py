"""Core probability and calibration utilities used by the PHAS0077 project."""

from __future__ import annotations

import numpy as np


def predictive_entropy(probabilities, eps: float = 1e-12):
    """Return predictive entropy using the natural logarithm.

    Parameters
    ----------
    probabilities : array-like
        One probability vector of shape (C,) or a batch of shape (N, C).
    eps : float
        Small numerical constant used to avoid log(0).

    Returns
    -------
    float or numpy.ndarray
        Entropy value(s).
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim not in (1, 2):
        raise ValueError("probabilities must have shape (C,) or (N, C)")
    if np.any(probs < 0):
        raise ValueError("probabilities must be non-negative")
    row_sums = probs.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("each probability vector must sum to 1")
    return -np.sum(probs * np.log(probs + eps), axis=-1)


def prediction_confidence(probabilities):
    """Return the largest class probability for each prediction."""
    probs = np.asarray(probabilities, dtype=float)
    return np.max(probs, axis=-1)


def probability_margin(probabilities):
    """Return the Top-1 minus Top-2 probability margin."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape[-1] < 2:
        raise ValueError("at least two classes are required")
    sorted_probs = np.sort(probs, axis=-1)
    return sorted_probs[..., -1] - sorted_probs[..., -2]


def multiclass_brier_score(y_true, probabilities):
    """Return the mean multiclass Brier score."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape (N, C)")
    if len(y_true) != len(probs):
        raise ValueError("y_true and probabilities must contain the same number of samples")
    one_hot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def negative_log_likelihood(y_true, probabilities, eps: float = 1e-12):
    """Return mean multiclass negative log-likelihood."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps)))


def expected_calibration_error(y_true, probabilities, n_bins: int = 10):
    """Compute equal-width expected calibration error."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if np.any(mask):
            ece += mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean())
    return float(ece)
