import numpy as np
import pytest

from src.baselines import bias_baseline, drug_mean
from src.knn_cf import KNNCF
from src.masking import split_observed
from src.metrics import rmse


def _toy(seed=0, n=60, m=25, missing=0.2):
    """Low-rank matrix with row + column effects and some missing cells."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, 2))
    V = rng.normal(size=(m, 2))
    M = 0.8 + 0.05 * (U @ V.T) + rng.normal(0, 0.01, (n, 1)) + rng.normal(0, 0.05, (1, m))
    M[rng.random(M.shape) < missing] = np.nan
    return M


def test_split_is_disjoint_and_only_hides_observed():
    M = _toy()
    s = split_observed(M, 0.1, 0.1, seed=1)
    test = {tuple(x) for x in s.test_idx}
    val = {tuple(x) for x in s.val_idx}
    assert not test & val
    assert all(not np.isnan(M[r, c]) for r, c in test | val)
    assert all(np.isnan(s.train[r, c]) for r, c in test | val)
    n_obs = (~np.isnan(M)).sum()
    assert (~np.isnan(s.train)).sum() == n_obs - len(test) - len(val)


def test_split_is_reproducible():
    M = _toy()
    a = split_observed(M, 0.1, 0.1, seed=7)
    b = split_observed(M, 0.1, 0.1, seed=7)
    assert np.array_equal(a.test_idx, b.test_idx)


def test_train_plus_val_restores_only_val():
    M = _toy()
    s = split_observed(M, 0.1, 0.1, seed=3)
    tv = s.train_plus_val()
    r, c = s.val_idx.T
    assert np.allclose(tv[r, c], M[r, c])
    r, c = s.test_idx.T
    assert np.isnan(tv[r, c]).all()


def test_no_leakage_prediction_ignores_hidden_value():
    """Changing a held-out true value must not change its prediction."""
    M = _toy()
    s = split_observed(M, 0.1, 0.0, seed=2)
    r, c = s.test_idx.T
    for kind in ("user", "item"):
        p1 = KNNCF(kind=kind, k=10).fit(s.train).predict(r, c)
        M2 = M.copy()
        M2[r, c] = 0.0
        s2 = split_observed(M2, 0.1, 0.0, seed=2)
        p2 = KNNCF(kind=kind, k=10).fit(s2.train).predict(r, c)
        assert np.allclose(p1, p2)


def test_duplicate_row_is_perfect_neighbour():
    """A cell line that is an exact copy of another should be predicted from it."""
    M = _toy(missing=0.0)
    M[1] = M[0]
    train = M.copy()
    train[1, 5] = np.nan
    m = KNNCF(kind="user", similarity="cosine", k=1, shrinkage=0, baseline="none", clip=None)
    m.fit(train)
    assert np.isclose(m.predict(np.array([1]), np.array([5]))[0], M[0, 5])


def test_knn_beats_baselines_on_structured_data():
    M = _toy(n=120, m=40)
    s = split_observed(M, 0.15, 0.0, seed=0)
    r, c = s.test_idx.T
    y = s.truth(s.test_idx)
    base = rmse(y, bias_baseline(s.train)[r, c])
    for kind in ("user", "item"):
        knn = rmse(y, KNNCF(kind=kind, k=20).fit(s.train).predict(r, c))
        assert knn < base
        assert knn < rmse(y, drug_mean(s.train)[r, c])


def test_multi_k_matches_single_k():
    M = _toy()
    s = split_observed(M, 0.1, 0.0, seed=4)
    r, c = s.test_idx.T
    m = KNNCF(kind="user", k=7).fit(s.train)
    assert np.allclose(m.predict(r, c), m.predict(r, c, ks=[3, 7, 50])[7])


def test_similarity_bounded_and_symmetric():
    M = _toy()
    for sim in ("cosine", "pearson"):
        S = KNNCF(similarity=sim).fit(M).S_.copy()
        np.fill_diagonal(S, 0)
        assert np.allclose(S, S.T)
        assert np.all(np.abs(S) <= 1 + 1e-9)


def test_entity_with_no_candidates_falls_back_to_baseline():
    M = _toy(missing=0.0)
    train = M.copy()
    train[:, 3] = np.nan      # nobody measured drug 3
    m = KNNCF(kind="user").fit(train)
    p = m.predict(np.array([0, 1]), np.array([3, 3]))
    assert np.allclose(p, np.clip(bias_baseline(train)[[0, 1], 3], 0, 1))


def test_bad_args():
    with pytest.raises(ValueError):
        KNNCF(kind="drug")
    with pytest.raises(ValueError):
        split_observed(_toy(), 0.6, 0.5, seed=0)
