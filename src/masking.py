"""Hold out random observed entries of the response matrix.

The matrix is cell lines (rows) x drugs (columns); NaN means "never measured".
Only entries that were actually measured can be hidden and scored, so every
split samples from the observed cells, never from the NaNs.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    train: np.ndarray        # matrix with val + test entries set to NaN
    val_idx: np.ndarray      # (n, 2) array of (row, col) held out for tuning
    test_idx: np.ndarray     # (n, 2) array of (row, col) held out for reporting
    full: np.ndarray         # the original matrix, used to read true values back

    def train_plus_val(self) -> np.ndarray:
        """Training matrix with the validation entries put back (used for the final test run)."""
        out = self.train.copy()
        r, c = self.val_idx[:, 0], self.val_idx[:, 1]
        out[r, c] = self.full[r, c]
        return out

    def truth(self, idx: np.ndarray) -> np.ndarray:
        return self.full[idx[:, 0], idx[:, 1]]


def split_observed(M: np.ndarray, test_frac: float, val_frac: float, seed: int) -> Split:
    """Randomly hide `test_frac` of observed entries for test and `val_frac` for validation.

    Fractions are of the *observed* entries. Test and validation never overlap.
    """
    if test_frac + val_frac >= 1:
        raise ValueError("test_frac + val_frac must be < 1")
    rng = np.random.default_rng(seed)
    obs = np.argwhere(~np.isnan(M))
    perm = rng.permutation(len(obs))
    n_test = int(round(test_frac * len(obs)))
    n_val = int(round(val_frac * len(obs)))
    test_idx = obs[perm[:n_test]]
    val_idx = obs[perm[n_test:n_test + n_val]]

    train = M.copy()
    train[test_idx[:, 0], test_idx[:, 1]] = np.nan
    train[val_idx[:, 0], val_idx[:, 1]] = np.nan
    return Split(train=train, val_idx=val_idx, test_idx=test_idx, full=M)
