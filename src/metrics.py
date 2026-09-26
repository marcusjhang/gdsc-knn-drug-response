import numpy as np
from scipy.stats import pearsonr, spearmanr

# AUC below this is treated as "the drug actually hits this cell line".
# Most AUCs sit near 1 (no effect), so a model can get a low overall RMSE by
# predicting ~0.95 everywhere; RMSE on this subset shows whether it gets the
# interesting cases right.
SENSITIVE_AUC = 0.8


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def per_row_spearman(rows, y, p, min_items=5):
    """Mean Spearman correlation between true and predicted AUC within each cell line.

    Measures whether the held-out drugs are put in the right order for a given
    cell line, which is what a drug ranking would use.
    """
    vals = []
    for r in np.unique(rows):
        m = rows == r
        if m.sum() >= min_items and np.std(y[m]) > 0 and np.std(p[m]) > 0:
            vals.append(spearmanr(y[m], p[m]).statistic)
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(y, p, rows):
    y = np.asarray(y)
    p = np.asarray(p)
    sens = y < SENSITIVE_AUC
    return {
        "rmse": rmse(y, p),
        "mae": mae(y, p),
        "pearson_r": float(pearsonr(y, p).statistic),
        "rmse_sensitive": rmse(y[sens], p[sens]),
        "rmse_insensitive": rmse(y[~sens], p[~sens]),
        "per_cell_line_spearman": per_row_spearman(rows, y, p),
        "n": int(len(y)),
    }
