"""Small reusable utilities for fixed-grid regional occlusion."""

from __future__ import annotations

import torch


def mask_grid_cell(image_tensor, row_idx: int, col_idx: int, grid_size: int = 7,
                   fill_value: float = 0.0):
    """Return a copy of an image tensor with one grid cell replaced.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Tensor of shape (N, C, H, W).
    row_idx, col_idx : int
        Zero-based grid coordinates.
    grid_size : int
        Number of rows and columns in the square grid.
    fill_value : float
        Replacement value in normalised tensor space.
    """
    if image_tensor.ndim != 4:
        raise ValueError("image_tensor must have shape (N, C, H, W)")
    if not (0 <= row_idx < grid_size and 0 <= col_idx < grid_size):
        raise ValueError("grid coordinates are outside the selected grid")

    masked = image_tensor.clone()
    _, _, height, width = masked.shape
    cell_h = height // grid_size
    cell_w = width // grid_size

    h1 = row_idx * cell_h
    h2 = (row_idx + 1) * cell_h if row_idx < grid_size - 1 else height
    w1 = col_idx * cell_w
    w2 = (col_idx + 1) * cell_w if col_idx < grid_size - 1 else width

    masked[:, :, h1:h2, w1:w2] = fill_value
    return masked
