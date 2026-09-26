"""Neighbourhood (KNN) collaborative filtering for the cell line x drug AUC matrix.

Two flavours, same maths on a transposed matrix:
  * kind="user": cell-line-based. To predict AUC(cell i, drug j), find the k cell
    lines most similar to i that *were* measured on drug j and average their AUC
    on j, weighted by similarity.
  * kind="item": drug-based. Find the k drugs most similar to j that cell line i
    *was* measured on and average i's AUC on them.

Prediction (Koren 2010, "baseline + neighbourhood"):
    r_hat(i, j) = b(i, j) + sum_v s(i, v) * (r(v, j) - b(v, j)) / sum_v |s(i, v)|
where b is a baseline (by default mu + b_cell + b_drug). Working on residuals
means a neighbour contributes "how much more or less sensitive than expected"
rather than a raw AUC, which removes the fact that some drugs are simply more
toxic than others. With baseline="none" this reduces to the plain weighted
average of neighbours' raw AUC.

Similarities are computed only over co-observed entries and shrunk toward 0
when the overlap is small: s <- s * n_common / (n_common + shrinkage).
"""
import numpy as np

from .baselines import bias_baseline, drug_mean

BASELINES = ("bias", "drug_mean", "none")
SIMILARITIES = ("cosine", "pearson")


class KNNCF:
    def __init__(self, kind="user", similarity="pearson", k=20, shrinkage=10.0,
                 baseline="bias", clip=(0.0, 1.0)):
        if kind not in ("user", "item"):
            raise ValueError(f"kind must be 'user' or 'item', got {kind!r}")
        if similarity not in SIMILARITIES:
            raise ValueError(f"similarity must be one of {SIMILARITIES}")
        if baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")
        self.kind = kind
        self.similarity = similarity
        self.k = k
        self.shrinkage = shrinkage
        self.baseline = baseline
        self.clip = clip

    # ------------------------------------------------------------------ fit
    def fit(self, train: np.ndarray) -> "KNNCF":
        # Fallback when no usable neighbour exists: always the bias baseline,
        # so baseline="none" does not fall back to predicting 0.
        fallback = bias_baseline(train)
        if self.baseline == "bias":
            offset = fallback
        elif self.baseline == "drug_mean":
            offset = drug_mean(train)
        else:
            offset = np.zeros_like(train)

        # Internally rows are always the entities we find neighbours among.
        A, offset, fallback = (train, offset, fallback) if self.kind == "user" \
            else (train.T, offset.T, fallback.T)
        self._W = ~np.isnan(A)
        self._R = np.where(self._W, A - offset, 0.0)   # residuals, 0 where unknown
        self._offset = offset
        self._fallback = fallback
        self.S_ = self._similarity(self._R, self._W)
        return self

    def _similarity(self, R, W):
        Wf = W.astype(float)
        X = R
        if self.similarity == "pearson":
            # centre each row on its own mean residual over observed entries
            n = Wf.sum(1, keepdims=True)
            mean = X.sum(1, keepdims=True) / np.maximum(n, 1)
            X = np.where(W, X - mean, 0.0)
        num = X @ X.T
        X2 = X * X
        norm_i = X2 @ Wf.T            # [i, v]: sum of x_i^2 over items both observed
        norm_v = norm_i.T              # [i, v]: sum of x_v^2 over the same items
        den = np.sqrt(norm_i * norm_v)
        S = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        if self.shrinkage > 0:
            n_common = Wf @ Wf.T
            S *= n_common / (n_common + self.shrinkage)
        np.fill_diagonal(S, -np.inf)   # never your own neighbour
        return S

    # -------------------------------------------------------------- predict
    def predict(self, rows: np.ndarray, cols: np.ndarray, ks=None):
        """Predict AUC for (rows[n], cols[n]) in the original cell-line x drug orientation.

        If `ks` is a list, returns {k: predictions} computed from one neighbour
        search (cheap way to sweep k). Otherwise uses self.k and returns an array.
        """
        single = ks is None
        ks = [self.k] if single else list(ks)
        q_ent, q_ctx = (rows, cols) if self.kind == "user" else (cols, rows)
        out = {k: np.empty(len(rows)) for k in ks}
        kmax = max(ks)

        for j in np.unique(q_ctx):
            sel = np.flatnonzero(q_ctx == j)
            ents = q_ent[sel]
            cand = np.flatnonzero(self._W[:, j])          # entities measured on context j
            base = self._offset[ents, j]
            fb = self._fallback[ents, j]
            if len(cand) == 0:
                for k in ks:
                    out[k][sel] = fb
                continue
            sims = self.S_[np.ix_(ents, cand)]
            kk = min(kmax, len(cand))
            top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
            s_top = np.take_along_axis(sims, top, 1)
            order = np.argsort(-s_top, axis=1)             # most similar first
            top = np.take_along_axis(top, order, 1)
            s_top = np.take_along_axis(s_top, order, 1)
            r_top = self._R[cand[top], j]
            # only positively similar neighbours vote; the rest are not "like" i
            s_use = np.where(s_top > 0, s_top, 0.0)
            num = np.cumsum(s_use * r_top, axis=1)
            den = np.cumsum(s_use, axis=1)
            for k in ks:
                c = min(k, kk) - 1
                ok = den[:, c] > 0
                pred = np.where(ok, base + num[:, c] / np.where(ok, den[:, c], 1.0), fb)
                out[k][sel] = pred

        if self.clip is not None:
            out = {k: np.clip(v, *self.clip) for k, v in out.items()}
        return out[self.k] if single else out
