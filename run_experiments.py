"""Mask random AUC values, predict them with KNN collaborative filtering, score with RMSE.

Protocol (per seed):
  1. Hide 10% of observed entries as TEST and another 10% as VALIDATION.
  2. Fit every KNN configuration on the remaining 80% and score it on VALIDATION.
  3. Pick the configuration with the lowest validation RMSE.
  4. Refit on TRAIN + VALIDATION and score once on TEST (never used for choices).
Repeated for 5 seeds; tables report mean +/- std across seeds.

Then: a sweep over how much of the matrix is masked, and an error analysis.

Usage:  .venv/bin/python run_experiments.py                 # full run (~25 min on a laptop)
        .venv/bin/python run_experiments.py --figures-only  # redraw figures from results/*.csv
Outputs go to results/ (CSV + JSON + summary.md) and figures/ (PNG).
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from src import baselines
from src.knn_cf import KNNCF
from src.masking import split_observed
from src.metrics import SENSITIVE_AUC, evaluate, rmse

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "cancer_cell_line_drug_auc_matrix.csv"
RES = ROOT / "results"
FIG = ROOT / "figures"

SEEDS = [0, 1, 2, 3, 4]
TEST_FRAC = 0.10
VAL_FRAC = 0.10
KS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200]
GRID = {
    "kind": ["user", "item"],
    "similarity": ["pearson", "cosine"],
    "shrinkage": [0, 10, 25, 50, 100, 200, 400, 800],
    "baseline": ["bias", "drug_mean", "none"],
}
IMPUTER_KS = [5, 10, 20, 40]
MASK_FRACS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
MASK_SEEDS = [0, 1, 2]

BASELINES = {
    "Global mean": baselines.global_mean,
    "Cell-line mean": baselines.cell_line_mean,
    "Drug mean": baselines.drug_mean,
    "Cell-line + drug bias": baselines.bias_baseline,
}


def load():
    df = pd.read_csv(DATA, index_col=0)
    df.columns = df.columns.astype(str)
    return df, df.values.astype(float)


def describe(df, M):
    W = ~np.isnan(M)
    vals = M[W]
    per_row = W.sum(1)
    per_col = W.sum(0)
    return {
        "n_cell_lines": int(M.shape[0]),
        "n_drugs": int(M.shape[1]),
        "n_observed": int(W.sum()),
        "missing_fraction": float(1 - W.mean()),
        "auc_mean": float(vals.mean()),
        "auc_median": float(np.median(vals)),
        "auc_std": float(vals.std()),
        "auc_min": float(vals.min()),
        "auc_max": float(vals.max()),
        "frac_auc_below_0.8": float((vals < SENSITIVE_AUC).mean()),
        "frac_auc_above_0.9": float((vals > 0.9).mean()),
        "obs_per_cell_line_min": int(per_row.min()),
        "obs_per_cell_line_median": float(np.median(per_row)),
        "obs_per_drug_min": int(per_col.min()),
        "obs_per_drug_median": float(np.median(per_col)),
        "cell_lines_with_<50_obs": int((per_row < 50).sum()),
        "duplicate_cell_lines": int(df.index.duplicated().sum()),
        "duplicate_drugs": int(df.columns.duplicated().sum()),
    }


def config_name(cfg):
    who = "Cell-line KNN" if cfg["kind"] == "user" else "Drug KNN"
    return f"{who} ({cfg['similarity']}, shrink={cfg['shrinkage']}, baseline={cfg['baseline']}, k={cfg['k']})"


def grid_search(train, idx, y):
    """Score every (config, k) on the given held-out entries. Returns a DataFrame."""
    r, c = idx.T
    rows = []
    for kind, sim, shrink, base in itertools.product(*GRID.values()):
        m = KNNCF(kind=kind, similarity=sim, shrinkage=shrink, baseline=base).fit(train)
        for k, p in m.predict(r, c, ks=KS).items():
            rows.append({"kind": kind, "similarity": sim, "shrinkage": shrink,
                         "baseline": base, "k": k, "rmse": rmse(y, p)})
    return pd.DataFrame(rows)


def imputer_predict(train, idx, k):
    """sklearn KNNImputer on the raw matrix: the 'simplest possible' KNN, for reference."""
    filled = KNNImputer(n_neighbors=k, weights="distance").fit_transform(train)
    return filled[idx[:, 0], idx[:, 1]]


def fit_predict(cfg, train, idx):
    m = KNNCF(kind=cfg["kind"], similarity=cfg["similarity"], shrinkage=cfg["shrinkage"],
              baseline=cfg["baseline"], k=cfg["k"]).fit(train)
    return m.predict(idx[:, 0], idx[:, 1])


def best_of(grid, **filters):
    g = grid
    for key, v in filters.items():
        g = g[g[key] == v]
    row = g.loc[g["rmse"].idxmin()]
    return {k: (row[k].item() if hasattr(row[k], "item") else row[k])
            for k in ["kind", "similarity", "shrinkage", "baseline", "k"]}


# ---------------------------------------------------------------- main study
def main_study(M):
    all_grids, test_rows, chosen, preds_seed0 = [], [], [], None
    for seed in SEEDS:
        t0 = time.time()
        s = split_observed(M, TEST_FRAC, VAL_FRAC, seed)
        yv = s.truth(s.val_idx)
        yt = s.truth(s.test_idx)
        tr_val = s.train_plus_val()
        rows_t = s.test_idx[:, 0]

        grid = grid_search(s.train, s.val_idx, yv).assign(seed=seed)
        all_grids.append(grid)

        # model selection on validation only
        best = best_of(grid)
        best_user = best_of(grid, kind="user")
        best_item = best_of(grid, kind="item")
        best_plain = best_of(grid, kind="user", similarity="cosine", shrinkage=0, baseline="none")
        imp_val = {k: rmse(yv, imputer_predict(s.train, s.val_idx, k)) for k in IMPUTER_KS}
        imp_k = min(imp_val, key=imp_val.get)
        chosen.append({"seed": seed, **best, "imputer_k": imp_k})

        def record(name, p, **extra):
            test_rows.append({"seed": seed, "method": name, **evaluate(yt, p, rows_t), **extra})
            return p

        # final test scores: refit on train + validation
        for name, fn in BASELINES.items():
            record(name, fn(tr_val)[s.test_idx[:, 0], s.test_idx[:, 1]])
        record("sklearn KNNImputer (raw AUC)", imputer_predict(tr_val, s.test_idx, imp_k), k=imp_k)
        record("Cell-line KNN, plain cosine, no baseline", fit_predict(best_plain, tr_val, s.test_idx),
               k=best_plain["k"])
        p_user = record("Cell-line KNN (tuned)", fit_predict(best_user, tr_val, s.test_idx), k=best_user["k"])
        p_item = record("Drug KNN (tuned)", fit_predict(best_item, tr_val, s.test_idx), k=best_item["k"])
        p_best = record("Best KNN on validation", fit_predict(best, tr_val, s.test_idx), k=best["k"])
        record("Blend: 0.5 cell-line KNN + 0.5 drug KNN", 0.5 * p_user + 0.5 * p_item)

        if seed == SEEDS[0]:
            preds_seed0 = {"split": s, "pred": p_best, "cfg": best}
        print(f"seed {seed}: best = {config_name(best)}  "
              f"test RMSE = {rmse(yt, p_best):.4f}  ({time.time() - t0:.0f}s)")

    return pd.concat(all_grids), pd.DataFrame(test_rows), pd.DataFrame(chosen), preds_seed0


# ------------------------------------------------------------ mask-fraction sweep
def mask_sweep(M, cfg):
    rows = []
    for frac, seed in itertools.product(MASK_FRACS, MASK_SEEDS):
        s = split_observed(M, frac, 0.0, seed)
        r, c = s.test_idx.T
        y = s.truth(s.test_idx)
        preds = {
            "Drug mean": baselines.drug_mean(s.train)[r, c],
            "Cell-line + drug bias": baselines.bias_baseline(s.train)[r, c],
            "Best KNN": fit_predict(cfg, s.train, s.test_idx),
        }
        for name, p in preds.items():
            rows.append({"mask_frac": frac, "seed": seed, "method": name, "rmse": rmse(y, p)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ error analysis
def error_analysis(df, info):
    s, p = info["split"], info["pred"]
    r, c = s.test_idx.T
    y = s.truth(s.test_idx)
    base = baselines.bias_baseline(s.train_plus_val())[r, c]
    n_obs_row = (~np.isnan(s.train_plus_val())).sum(1)[r]
    n_obs_col = (~np.isnan(s.train_plus_val())).sum(0)[c]
    t = pd.DataFrame({"cell_line": df.index[r], "drug_id": df.columns[c], "true": y,
                      "pred_knn": p, "pred_bias": base,
                      "cell_line_train_obs": n_obs_row, "drug_train_obs": n_obs_col})
    t["sq_err_knn"] = (t.pred_knn - t.true) ** 2
    t["sq_err_bias"] = (t.pred_bias - t.true) ** 2

    def by(col, bins, labels):
        g = t.groupby(pd.cut(t[col], bins=bins, labels=labels), observed=True)
        return g.agg(n=("true", "size"),
                     rmse_knn=("sq_err_knn", lambda v: np.sqrt(v.mean())),
                     rmse_bias=("sq_err_bias", lambda v: np.sqrt(v.mean())))

    by_obs = by("cell_line_train_obs", [0, 50, 150, 250, 300],
                ["<50", "50-149", "150-249", "250+"])
    by_auc = by("true", [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
                ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1"])
    per_drug = (t.groupby("drug_id")
                 .agg(n=("true", "size"), true_auc_std=("true", "std"),
                      rmse_knn=("sq_err_knn", lambda v: np.sqrt(v.mean())),
                      rmse_bias=("sq_err_bias", lambda v: np.sqrt(v.mean())))
                 .query("n >= 20").sort_values("rmse_knn", ascending=False))
    return t, by_obs, by_auc, per_drug


# ------------------------------------------------------------------------ figures
# Categorical slots from the reference palette (light mode), fixed order.
C_KNN, C_ALT, C_BASE, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID_C, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID_C)
    ax.tick_params(colors=INK2)


def make_figures(summary, grid, sweep, t, by_obs, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})

    # 1. test RMSE by method
    s = summary.sort_values("rmse_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [C_KNN if ("KNN" in m or "Blend" in m) else C_BASE for m in s.index]
    ax.barh(s.index, s["rmse_mean"], xerr=s["rmse_std"], color=colors, height=0.6,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    for i, v in enumerate(s["rmse_mean"]):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", color=INK, fontsize=9)
    ax.invert_yaxis()
    _style(ax, "Test RMSE on masked AUC values (lower is better; mean ± sd over 5 seeds)",
           "RMSE", "")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "01_test_rmse_by_method.png"); plt.close(fig)

    # 2. validation RMSE vs k, best setting per kind
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind, color, label in [("user", C_KNN, "Cell-line KNN"), ("item", C_ALT, "Drug KNN")]:
        g = grid[grid.kind == kind]
        cfg = g.groupby(["similarity", "shrinkage", "baseline"])["rmse"].min().idxmin()
        g = g[(g.similarity == cfg[0]) & (g.shrinkage == cfg[1]) & (g.baseline == cfg[2])]
        curve = g.groupby("k")["rmse"].agg(["mean", "std"])
        ax.plot(curve.index, curve["mean"], color=color, lw=2, marker="o", ms=4,
                label=f"{label} ({cfg[0]}, shrink={cfg[1]}, baseline={cfg[2]})")
        ax.fill_between(curve.index, curve["mean"] - curve["std"], curve["mean"] + curve["std"],
                        color=color, alpha=0.15, lw=0)
    ax.set_xscale("log")
    ticks = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    ax.set_xticks(ticks, [str(k) for k in ticks])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Validation RMSE vs number of neighbours k", "k (log scale)", "Validation RMSE")
    fig.savefig(FIG / "02_validation_rmse_vs_k.png"); plt.close(fig)

    # 3. effect of baseline / similarity / shrinkage (best k each)
    factors = ["baseline", "similarity", "shrinkage"]
    aggs = {}
    for factor in factors:
        best = grid.groupby(["seed", "kind", factor])["rmse"].min().reset_index()
        aggs[factor] = best.groupby(["kind", factor])["rmse"].mean().unstack(0)
    lo = min(np.nanmin(a.values) for a in aggs.values())
    hi = max(np.nanmax(a.values) for a in aggs.values())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, factor in zip(axes, factors):
        agg = aggs[factor]
        x = np.arange(len(agg))
        ax.bar(x - 0.2, agg["user"], 0.38, color=C_KNN, label="Cell-line KNN")
        ax.bar(x + 0.2, agg["item"], 0.38, color=C_ALT, label="Drug KNN")
        ax.set_xticks(x, [str(v) for v in agg.index])
        ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.1 * (hi - lo))
        _style(ax, f"By {factor}", factor, "Best validation RMSE" if factor == "baseline" else "")
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Ablation: each bar is the best validation RMSE over all other settings "
                 "(mean of 5 seeds; y-axis zoomed)", x=0.01, y=1.04, ha="left", fontsize=11, color=INK)
    fig.savefig(FIG / "03_ablation.png"); plt.close(fig)

    # 4. predicted vs true
    fig, ax = plt.subplots(figsize=(5, 5))
    hb = ax.hexbin(t["true"], t["pred_knn"], gridsize=60, bins="log", cmap="Blues", mincnt=1)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    fig.colorbar(hb, ax=ax, label="count (log)")
    _style(ax, "Best KNN: predicted vs true AUC (test, seed 0)", "True AUC", "Predicted AUC")
    fig.savefig(FIG / "04_pred_vs_true.png"); plt.close(fig)

    # 5. mask-fraction sweep
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in [("Best KNN", C_KNN), ("Cell-line + drug bias", C_BASE), ("Drug mean", C_REF)]:
        g = sweep[sweep.method == name].groupby("mask_frac")["rmse"].agg(["mean", "std"])
        ax.errorbar(g.index * 100, g["mean"], yerr=g["std"], color=color, lw=2, marker="o",
                    ms=5, capsize=2, label=name)
    ax.legend(frameon=False)
    _style(ax, "Test RMSE as more of the matrix is masked", "% of observed values masked", "Test RMSE")
    fig.savefig(FIG / "05_rmse_vs_mask_fraction.png"); plt.close(fig)

    # 6. error by how many drugs a cell line has in training
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(by_obs))
    ax.bar(x - 0.2, by_obs["rmse_knn"], 0.38, color=C_KNN, label="Best KNN")
    ax.bar(x + 0.2, by_obs["rmse_bias"], 0.38, color=C_BASE, label="Cell-line + drug bias")
    ax.set_xticks(x, [f"{i}\n(n={n})" for i, n in zip(by_obs.index, by_obs["n"])])
    ax.legend(frameon=False)
    _style(ax, "Test RMSE by how many drugs the cell line was measured on",
           "Observed drugs for that cell line in training", "Test RMSE")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "06_rmse_by_cell_line_coverage.png"); plt.close(fig)

    # 7. AUC distribution (why RMSE looks small)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(M[~np.isnan(M)], bins=80, color=C_KNN)
    ax.axvline(SENSITIVE_AUC, color=INK2, ls="--", lw=1)
    ax.text(SENSITIVE_AUC - 0.01, ax.get_ylim()[1] * 0.9, f"AUC < {SENSITIVE_AUC}\n= \"sensitive\" (our cutoff)",
            ha="right", color=INK2, fontsize=9)
    _style(ax, "Distribution of all observed AUC values", "AUC", "Count")
    fig.savefig(FIG / "07_auc_distribution.png"); plt.close(fig)


# ---------------------------------------------------------------------------- main
def fmt(m, s):
    return f"{m:.4f} ± {s:.4f}"


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df, M = load()
    info = describe(df, M)
    (RES / "data_summary.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

    grid, test, chosen, seed0 = main_study(M)
    grid.to_csv(RES / "validation_grid.csv", index=False)
    test.to_csv(RES / "test_results_per_seed.csv", index=False)
    chosen.to_csv(RES / "chosen_config_per_seed.csv", index=False)

    metrics = ["rmse", "mae", "pearson_r", "rmse_sensitive", "rmse_insensitive",
               "per_cell_line_spearman"]
    summary = test.groupby("method")[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.sort_values("rmse_mean")
    summary.to_csv(RES / "test_results_summary.csv")

    # one config for the follow-up studies: the most frequently chosen one
    key = ["kind", "similarity", "shrinkage", "baseline", "k"]
    mode = chosen.groupby(key).size().idxmax()
    cfg = dict(zip(key, [v.item() if hasattr(v, "item") else v for v in mode]))
    print("config used for sweep/analysis:", config_name(cfg))

    sweep = mask_sweep(M, cfg)
    sweep.to_csv(RES / "mask_fraction_sweep.csv", index=False)

    t, by_obs, by_auc, per_drug = error_analysis(df, seed0)
    t.to_csv(RES / "test_predictions_seed0.csv", index=False)
    by_obs.to_csv(RES / "error_by_cell_line_coverage.csv")
    by_auc.to_csv(RES / "error_by_true_auc.csv")
    per_drug.to_csv(RES / "error_by_drug.csv")

    make_figures(summary, grid, sweep, t, by_obs, M)

    # ------------------------------------------------------------ summary.md
    lines = ["# KNN imputation of masked AUC values: results", "",
             "Generated by `run_experiments.py`. All numbers are on held-out TEST entries that were "
             "never used for choosing settings.", "",
             "## Data", ""]
    lines += [f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in info.items()]
    lines += ["", f"## Test results ({len(SEEDS)} seeds, {int(TEST_FRAC*100)}% test / "
              f"{int(VAL_FRAC*100)}% validation / rest train)", "",
              "| Method | RMSE | MAE | Pearson r | RMSE (AUC<0.8) | RMSE (AUC≥0.8) | Spearman within cell line |",
              "|---|---|---|---|---|---|---|"]
    for m, row in summary.iterrows():
        lines.append(f"| {m} | " + " | ".join(fmt(row[f'{x}_mean'], row[f'{x}_std']) for x in metrics) + " |")
    lines += ["", "## Configuration chosen on validation, per seed", "",
              chosen.to_markdown(index=False), "",
              f"Config used for the mask sweep and error analysis: **{config_name(cfg)}**", "",
              "## RMSE vs fraction masked (3 seeds)", "",
              sweep.groupby(["mask_frac", "method"])["rmse"].mean().unstack().round(4).to_markdown(), "",
              "## Error by how many drugs the cell line has in training (seed 0)", "",
              by_obs.round(4).to_markdown(), "",
              "## Error by true AUC (seed 0)", "",
              by_auc.round(4).to_markdown(), "",
              "## Hardest drugs for KNN (seed 0, ≥20 test entries)", "",
              per_drug.head(10).round(4).to_markdown(), ""]
    (RES / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def figures_only():
    """Redraw figures from the CSVs in results/ without re-running the experiments."""
    _, M = load()
    summary = pd.read_csv(RES / "test_results_summary.csv", index_col=0)
    grid = pd.read_csv(RES / "validation_grid.csv")
    sweep = pd.read_csv(RES / "mask_fraction_sweep.csv")
    t = pd.read_csv(RES / "test_predictions_seed0.csv")
    by_obs = pd.read_csv(RES / "error_by_cell_line_coverage.csv", index_col=0)
    make_figures(summary, grid, sweep, t, by_obs, M)


if __name__ == "__main__":
    import sys
    figures_only() if "--figures-only" in sys.argv else main()
