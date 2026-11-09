"""
California Housing - OLS vs Ridge (L2) vs Lasso (L1)
====================================================
load -> EDA -> impute -> feature engineering -> multicollinearity (VIF)
-> scale/encode -> GridSearchCV (Ridge, Lasso) -> evaluate -> diagnostics -> save

Run:  python src/pipeline.py
      python src/pipeline.py --drop-capped   # remove censored $500k rows
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from features import CAT, RAW_COUNTS, TARGET, Winsorizer, engineer_features  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)  # Lasso convergence chatter

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
FIG = ROOT / "reports" / "figures"
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Load
# --------------------------------------------------------------------------- #
def load_data(url: str = DATA_URL) -> pd.DataFrame:
    cache = ROOT / "data" / "housing.csv"
    try:
        df = pd.read_csv(url)
        df.to_csv(cache, index=False)
    except Exception as exc:
        print(f"[warn] download failed ({exc}); using cached copy")
        df = pd.read_csv(cache)
    return df


# --------------------------------------------------------------------------- #
# 2. EDA
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame) -> None:
    print("\n=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    nulls = df.isna().sum()
    print("Missing values:\n", nulls[nulls > 0].to_string())
    capped = int((df[TARGET] >= 500001).sum())
    print(f"Rows with censored target (>= $500,001): {capped} ({capped / len(df):.1%})")

    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df[TARGET], bins=60, kde=True, ax=ax[0], color="#4c72b0")
    ax[0].set(title="Target distribution (note spike at $500k cap)", xlabel="median_house_value")
    sc = ax[1].scatter(df["longitude"], df["latitude"], c=df[TARGET], s=4, cmap="viridis", alpha=0.6)
    ax[1].set(title="Geography of house values", xlabel="longitude", ylabel="latitude")
    plt.colorbar(sc, ax=ax[1], label="median_house_value")
    plt.tight_layout()
    plt.savefig(FIG / "01_target_and_geography.png", dpi=130)
    plt.close()

    corr = df.select_dtypes("number").corr()
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.8})
    plt.title("Correlation heatmap - raw numeric features")
    plt.tight_layout()
    plt.savefig(FIG / "02_correlation_raw.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# 3. Multicollinearity (feature engineering lives in features.py)
# --------------------------------------------------------------------------- #
def vif_table(X: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 - R^2_j), regressing feature j on all the others (no extra dependency)."""
    X = (X - X.mean()) / X.std()
    vifs = {}
    for col in X.columns:
        others = X.drop(columns=col)
        r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
        vifs[col] = 1.0 / max(1e-12, 1.0 - r2)
    return pd.Series(vifs).sort_values(ascending=False)


def multicollinearity_report(df_fe: pd.DataFrame) -> dict[str, pd.Series]:
    num = df_fe.drop(columns=[TARGET] + CAT)
    num = num.fillna(num.median())
    inter = ["income_x_log_rooms", "income_x_age"]
    raw_cols = ["longitude", "latitude", "housing_median_age", "median_income"] + RAW_COUNTS
    ratio_cols = [c for c in num.columns if c not in RAW_COUNTS + inter]
    all_cols = [c for c in num.columns if c not in RAW_COUNTS]
    stages = {"1. Raw features": vif_table(num[raw_cols]),
              "2. Ratios replace raw counts": vif_table(num[ratio_cols]),
              "3. + interaction terms": vif_table(num[all_cols])}
    print("\n=== VIF (multicollinearity; > 10 is a red flag) ===")
    for k, v in stages.items():
        print(f"\n{k}\n", v.round(1).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for a_, (k, v) in zip(ax, stages.items()):
        v.sort_values().plot.barh(ax=a_, color="#dd8452")
        a_.axvline(10, ls="--", c="k", lw=1)
        a_.set(title=f"VIF - {k}", xlabel="VIF (dashed = 10)")
    plt.tight_layout()
    plt.savefig(FIG / "03_vif.png", dpi=130)
    plt.close()

    corr = num[all_cols].corr()
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    plt.title("Correlation heatmap - engineered features")
    plt.tight_layout()
    plt.savefig(FIG / "04_correlation_engineered.png", dpi=130)
    plt.close()
    pd.DataFrame(stages).round(2).to_csv(ROOT / "reports" / "vif.csv")
    return stages


# --------------------------------------------------------------------------- #
# 4. Models
# --------------------------------------------------------------------------- #
def make_preprocessor(num_cols: list[str]) -> ColumnTransformer:
    """Impute (median) -> winsorise (0.5/99.5%, train-fit) -> scale numerics; one-hot categoricals (drop first -> no dummy trap for OLS)."""
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("winsor", Winsorizer(0.005, 0.995)),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CAT),
    ], verbose_feature_names_out=False)


def make_pipeline(model, num_cols: list[str], interactions: bool) -> Pipeline:
    steps = [("prep", make_preprocessor(num_cols))]
    if interactions:  # all pairwise products of the (already scaled) features
        steps.append(("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
    steps.append(("model", model))
    return Pipeline(steps)


ALPHA_GRIDS = {
    "Ridge (L2)": np.logspace(-2, 5, 15),
    "Lasso (L1)": np.logspace(-1, 4, 11),
}


def fit_all(X_train, y_train, num_cols):
    cv = KFold(5, shuffle=True, random_state=SEED)
    fitted, searches = {}, {}
    for fs_name, inter in (("Engineered", False), ("Engineered + interactions", True)):
        # OLS (no hyper-parameters)
        ols = make_pipeline(LinearRegression(), num_cols, inter).fit(X_train, y_train)
        fitted[(fs_name, "OLS")] = ols
        # Ridge / Lasso with GridSearchCV over alpha
        for name, model in (("Ridge (L2)", Ridge(random_state=SEED)),
                            ("Lasso (L1)", Lasso(max_iter=3000, tol=1e-2, random_state=SEED))):
            gs = GridSearchCV(make_pipeline(model, num_cols, inter),
                              {"model__alpha": ALPHA_GRIDS[name]},
                              scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
            gs.fit(X_train, y_train)
            fitted[(fs_name, name)] = gs.best_estimator_
            searches[(fs_name, name)] = gs
            a = gs.best_params_["model__alpha"]
            edge = " (!! at grid edge - widen grid)" if a in (ALPHA_GRIDS[name][0], ALPHA_GRIDS[name][-1]) else ""
            print(f"[tune] {fs_name:26s} {name:11s} best alpha={a:10.3f}  CV RMSE={-gs.best_score_:,.0f}{edge}")
    return fitted, searches, cv


def evaluate(fitted, searches, X_train, y_train, X_test, y_test, cv) -> pd.DataFrame:
    rows = []
    for (fs, name), model in fitted.items():
        pred = model.predict(X_test)
        if (fs, name) in searches:  # reuse folds already computed by GridSearchCV
            gs = searches[(fs, name)]
            i = gs.best_index_
            cv_mean, cv_std = -gs.cv_results_["mean_test_score"][i], gs.cv_results_["std_test_score"][i]
        else:
            cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv,
                                       scoring="neg_root_mean_squared_error")
            cv_mean, cv_std = cv_rmse.mean(), cv_rmse.std()
        n_zero = None
        if name.startswith("Lasso"):
            n_zero = int((model.named_steps["model"].coef_ == 0).sum())
        n_feat = len(model.named_steps["model"].coef_)
        rows.append({
            "Feature set": fs, "Model": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred),
            "CV RMSE (mean)": cv_mean, "CV RMSE (std)": cv_std,
            "alpha": model.named_steps["model"].alpha if hasattr(model.named_steps["model"], "alpha") else np.nan,
            "# features": n_feat, "# zeroed by Lasso": n_zero,
        })
    return pd.DataFrame(rows).set_index(["Feature set", "Model"])


# --------------------------------------------------------------------------- #
# 5. Diagnostics plots
# --------------------------------------------------------------------------- #
def plot_pred_vs_actual(fitted, X_test, y_test, fs: str, fname: str) -> None:
    models = [m for (f, m) in fitted if f == fs]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    lim = [0, max(y_test.max(), 5e5) * 1.02]
    for ax, m in zip(axes, models):
        pred = fitted[(fs, m)].predict(X_test)
        ax.scatter(y_test, pred, s=6, alpha=0.35, color="#4c72b0")
        ax.plot(lim, lim, "r--", lw=1.5, label="perfect")
        ax.set(title=f"{m}  R²={r2_score(y_test, pred):.3f}  RMSE=${np.sqrt(mean_squared_error(y_test, pred)):,.0f}",
               xlabel="Actual ($)", xlim=lim, ylim=lim)
    axes[0].set_ylabel("Predicted ($)")
    plt.suptitle(f"Predicted vs Actual - {fs}", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=130)
    plt.close()


def plot_residuals(fitted, X_test, y_test, fs: str, fname: str) -> None:
    models = [m for (f, m) in fitted if f == fs]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, m in enumerate(models):
        pred = fitted[(fs, m)].predict(X_test)
        res = y_test.values - pred
        sns.histplot(res, bins=60, kde=True, ax=axes[0, j], color="#55a868")
        axes[0, j].axvline(0, c="r", ls="--")
        axes[0, j].set(title=f"{m}: residuals (mean={res.mean():,.0f}, sd={res.std():,.0f})", xlabel="Actual - Predicted ($)")
        axes[1, j].scatter(pred, res, s=6, alpha=0.3, color="#c44e52")
        axes[1, j].axhline(0, c="k", lw=1)
        axes[1, j].set(title=f"{m}: residuals vs fitted", xlabel="Predicted ($)", ylabel="Residual ($)")
    plt.suptitle(f"Residual diagnostics - {fs}", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=130)
    plt.close()


def plot_alpha_curves(searches) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, name in zip(axes, ("Ridge (L2)", "Lasso (L1)")):
        for fs, color in (("Engineered", "#4c72b0"), ("Engineered + interactions", "#dd8452")):
            gs = searches[(fs, name)]
            a = gs.cv_results_["param_model__alpha"].astype(float)
            m, s = -gs.cv_results_["mean_test_score"], gs.cv_results_["std_test_score"]
            ax.plot(a, m, "-o", ms=3, color=color, label=fs)
            ax.fill_between(a, m - s, m + s, alpha=0.15, color=color)
            ax.axvline(gs.best_params_["model__alpha"], color=color, ls=":")
        ax.set(xscale="log", title=f"{name}: CV RMSE vs alpha", xlabel="alpha (log)", ylabel="CV RMSE ($)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "07_alpha_curves.png", dpi=130)
    plt.close()


def plot_coefficients(fitted) -> None:
    fs = "Engineered"
    names = fitted[(fs, "OLS")][:-1].get_feature_names_out()
    coefs = pd.DataFrame({m: fitted[(fs, m)].named_steps["model"].coef_
                          for m in ("OLS", "Ridge (L2)", "Lasso (L1)")}, index=names)
    coefs.sort_values("OLS").plot.barh(figsize=(10, 6.5), width=0.8)
    plt.title("Standardised coefficients (Engineered features)")
    plt.xlabel("coefficient ($ per 1 SD)")
    plt.tight_layout()
    plt.savefig(FIG / "08_coefficients.png", dpi=130)
    plt.close()
    coefs.round(0).to_csv(ROOT / "reports" / "coefficients_engineered.csv")

    # Lasso on the interaction set: what survives?
    fs = "Engineered + interactions"
    names = fitted[(fs, "Lasso (L1)")][:-1].get_feature_names_out()
    c = pd.Series(fitted[(fs, "Lasso (L1)")].named_steps["model"].coef_, index=names)
    top = c[c != 0].reindex(c[c != 0].abs().sort_values(ascending=False).index).head(15)
    plt.figure(figsize=(10, 6))
    top.sort_values().plot.barh(color=np.where(top.sort_values() > 0, "#55a868", "#c44e52"))
    plt.title(f"Lasso + interactions: top 15 of {int((c != 0).sum())} surviving features "
              f"({int((c == 0).sum())} zeroed out)")
    plt.tight_layout()
    plt.savefig(FIG / "09_lasso_selected_interactions.png", dpi=130)
    plt.close()


def plot_benchmark(results: pd.DataFrame) -> None:
    r = results.reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric in zip(axes, ("RMSE", "MAE", "R2")):
        sns.barplot(data=r, x="Model", y=metric, hue="Feature set", ax=ax, palette="Set2")
        ax.set_title(metric + (" (lower is better)" if metric != "R2" else " (higher is better)"))
        if metric != "R2":
            lo = r[metric].min()
            ax.set_ylim(lo * 0.9, r[metric].max() * 1.04)
        else:
            ax.set_ylim(r[metric].min() - 0.03, r[metric].max() + 0.02)
        if ax is not axes[0]:
            ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig(FIG / "10_benchmark.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    df = load_data(args.url)
    run_eda(df)
    if args.drop_capped:
        df = df[df[TARGET] < 500001].reset_index(drop=True)
        print(f"[info] dropped capped rows -> {len(df)} rows remain")

    df_fe = engineer_features(df)
    multicollinearity_report(df_fe)

    num_cols = [c for c in df_fe.columns if c not in [TARGET] + CAT + RAW_COUNTS]
    X, y = df_fe[num_cols + CAT], df_fe[TARGET]

    # Stratify the split on income bands so train/test have the same income mix
    strata = pd.cut(df_fe["median_income"], [0, 1.5, 3, 4.5, 6, np.inf], labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=strata)
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}  | numeric features: {num_cols}")

    # Naive baseline for context
    base_rmse = np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean())))
    print(f"Baseline (predict the mean) RMSE: ${base_rmse:,.0f}")

    fitted, searches, cv = fit_all(X_train, y_train, num_cols)
    results = evaluate(fitted, searches, X_train, y_train, X_test, y_test, cv)
    pd.set_option("display.width", 220)
    print("\n=== FINAL BENCHMARK (test set, $) ===")
    print(results.round(3).to_string())

    for fs, tag in (("Engineered", "05"), ("Engineered + interactions", "06")):
        slug = "base" if fs == "Engineered" else "interactions"
        plot_pred_vs_actual(fitted, X_test, y_test, fs, f"{tag}a_pred_vs_actual_{slug}.png")
        plot_residuals(fitted, X_test, y_test, fs, f"{tag}b_residuals_{slug}.png")
    plot_alpha_curves(searches)
    plot_coefficients(fitted)
    plot_benchmark(results)

    results.round(4).to_csv(ROOT / "reports" / "benchmark_results.csv")
    best_key = results["RMSE"].idxmin()
    joblib.dump(fitted[best_key], ROOT / "models" / "best_model.joblib")
    (ROOT / "models" / "metadata.json").write_text(json.dumps({
        "best_feature_set": best_key[0], "best_model": best_key[1],
        "numeric_features": num_cols, "categorical_features": CAT,
        "baseline_rmse": base_rmse, "test_rmse": float(results.loc[best_key, "RMSE"]),
        "note": "Input must be passed through engineer_features() first.", "seed": SEED}, indent=2))
    print(f"\nBest by test RMSE: {best_key}  -> models/best_model.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DATA_URL)
    ap.add_argument("--drop-capped", action="store_true", help="remove censored $500k rows")
    main(ap.parse_args())
