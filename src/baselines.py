"""Non-personalised predictors every KNN result is compared against.

Each function takes a training matrix (NaN = unknown) and returns a full
prediction matrix of the same shape.
"""
import numpy as np


def global_mean(train: np.ndarray) -> np.ndarray:
    return np.full(train.shape, np.nanmean(train))


def drug_mean(train: np.ndarray) -> np.ndarray:
    """Predict each drug's average AUC over the cell lines it was measured on."""
    mu = np.nanmean(train)
    col = _nanmean_or(train, axis=0, fallback=mu)
    return np.broadcast_to(col, train.shape).copy()


def cell_line_mean(train: np.ndarray) -> np.ndarray:
    """Predict each cell line's average AUC over the drugs it was measured on."""
    mu = np.nanmean(train)
    row = _nanmean_or(train, axis=1, fallback=mu)
    return np.broadcast_to(row[:, None], train.shape).copy()


def bias_baseline(train: np.ndarray, reg_row: float = 5.0, reg_col: float = 5.0,
                  n_iter: int = 15) -> np.ndarray:
    """mu + b_cell + b_drug, fitted by alternating regularised least squares.

    The regularisers shrink a bias toward 0 when it is estimated from few
    observations, which matters here because some cell lines have only a
    dozen measured drugs.
    """
    W = ~np.isnan(train)
    X = np.where(W, train, 0.0)
    mu = X.sum() / W.sum()
    b_row = np.zeros(train.shape[0])
    b_col = np.zeros(train.shape[1])
    n_row = W.sum(1)
    n_col = W.sum(0)
    for _ in range(n_iter):
        b_col = ((X - mu - b_row[:, None]) * W).sum(0) / (reg_col + n_col)
        b_row = ((X - mu - b_col[None, :]) * W).sum(1) / (reg_row + n_row)
    return mu + b_row[:, None] + b_col[None, :]


def _nanmean_or(a, axis, fallback):
    W = ~np.isnan(a)
    n = W.sum(axis)
    s = np.where(W, a, 0.0).sum(axis)
    return np.where(n > 0, s / np.maximum(n, 1), fallback)
