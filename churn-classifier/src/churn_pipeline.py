"""
Telco Customer Churn - Random Forest vs XGBoost under class imbalance
=====================================================================
load -> clean -> EDA -> compare imbalance strategies (none / class weights / SMOTE)
-> tune with RandomizedSearchCV (ROC-AUC) -> choose decision threshold on OOF predictions
-> evaluate on held-out test -> ROC / PR curves, confusion matrices, feature importance.

Run:  python src/churn_pipeline.py            # full run (~5-8 min on 1 core)
      python src/churn_pipeline.py --quick     # fewer search iterations (~2 min)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, balanced_accuracy_score,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
SEED = 42
TARGET = "Churn"
NUM = ["tenure", "MonthlyCharges", "TotalCharges"]


# --------------------------------------------------------------------------- #
# 1-2. Load and clean
# --------------------------------------------------------------------------- #
def load_data(url: str = URL) -> pd.DataFrame:
    cache = ROOT / "data" / "telco_churn.csv"
    try:
        df = pd.read_csv(url)
        df.to_csv(cache, index=False)
    except Exception as exc:
        print(f"[warn] download failed ({exc}); using cached copy")
        df = pd.read_csv(cache)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns="customerID").copy()
    # TotalCharges is text because 11 rows hold a blank string. All 11 have tenure == 0
    # (brand-new customers, never billed) -> the correct value is 0, not a median.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_null = int(df["TotalCharges"].isna().sum())
    assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    print(f"[clean] TotalCharges: {n_null} blanks -> 0.0 (all had tenure == 0)")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})  # binary flag -> categorical
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame) -> None:
    rate = df[TARGET].mean()
    print(f"\n=== OVERVIEW ===\nRows: {len(df)} | churn rate: {rate:.1%} | imbalance ratio (stay:churn) = {(1-rate)/rate:.2f}:1")
    print(f"Duplicate feature rows: {int(df.duplicated().sum())} (kept: distinct customers can share a profile)")
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    df[TARGET].map({0: "Stayed", 1: "Churned"}).value_counts().plot.bar(ax=ax[0, 0], color=["#4c72b0", "#c44e52"], rot=0)
    ax[0, 0].set(title=f"Class balance (churn = {rate:.1%})", ylabel="customers")
    for key, col in (("Contract", ax[0, 1]), ("InternetService", ax[1, 0])):
        df.groupby(key)[TARGET].mean().sort_values().plot.barh(ax=col, color="#dd8452")
        col.set(title=f"Churn rate by {key}", xlabel="churn rate")
    sns.kdeplot(data=df, x="tenure", hue=df[TARGET].map({0: "Stayed", 1: "Churned"}), fill=True, common_norm=False, ax=ax[1, 1])
    ax[1, 1].set(title="Tenure (months) by outcome")
    plt.tight_layout()
    plt.savefig(FIG / "01_eda.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# 3-4. Models and imbalance strategies
# --------------------------------------------------------------------------- #
def make_preprocessor(cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], verbose_feature_names_out=False)


def make_clf(name: str, strategy: str, spw: float):
    weighted = strategy == "class weights"
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED,
                                      class_weight="balanced_subsample" if weighted else None)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8,
                             tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED,
                             scale_pos_weight=spw if weighted else 1.0)
    return LogisticRegression(max_iter=2000, class_weight="balanced" if weighted else None, random_state=SEED)


def build(name: str, strategy: str, cat_cols: list[str], spw: float) -> ImbPipeline:
    """SMOTE sits INSIDE the pipeline -> applied to training folds only, never to validation/test data."""
    steps = [("prep", make_preprocessor(cat_cols))]
    if strategy == "SMOTE":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", make_clf(name, strategy, spw)))
    return ImbPipeline(steps)


SPACES = {
    "Random Forest": {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [4, 6, 8, 12, None],
                      "clf__min_samples_leaf": [1, 3, 5, 10, 20], "clf__max_features": ["sqrt", 0.3, 0.5]},
    "XGBoost": {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [2, 3, 4, 5],
                "clf__learning_rate": [0.01, 0.02, 0.05, 0.1], "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 3, 5],
                "clf__reg_lambda": [1, 5, 10]},
}


def compare_strategies(X, y, cat_cols, spw, cv) -> pd.DataFrame:
    rows = []
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall", "precision": "precision", "f1": "f1"}
    for model in ("Logistic Regression", "Random Forest", "XGBoost"):
        for strat in ("none", "class weights", "SMOTE"):
            res = cross_validate(build(model, strat, cat_cols, spw), X, y, cv=cv, scoring=scoring, n_jobs=1)
            row = {"Model": model, "Strategy": strat}
            for k in scoring:
                row[k] = res[f"test_{k}"].mean()
            row["roc_auc_std"] = res["test_roc_auc"].std()
            rows.append(row)
            print(f"[cv] {model:19s} {strat:13s} AUC={row['roc_auc']:.4f}±{row['roc_auc_std']:.4f}  "
                  f"recall@0.5={row['recall']:.3f}  precision@0.5={row['precision']:.3f}")
    return pd.DataFrame(rows).set_index(["Model", "Strategy"])


def tune(model: str, strategy: str, X, y, cat_cols, spw, cv, n_iter: int):
    search = RandomizedSearchCV(build(model, strategy, cat_cols, spw), SPACES[model], n_iter=n_iter,
                                scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=1, refit=True)
    search.fit(X, y)
    print(f"[tune] {model} ({strategy}): best CV AUC={search.best_score_:.4f}")
    print("       ", {k.replace('clf__', ''): v for k, v in search.best_params_.items()})
    return search


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true, proba) -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.argmax(f1)])


def metrics_at(y_true, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    return {"threshold": round(thr, 3), "precision": precision_score(y_true, pred), "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred), "balanced_acc": balanced_accuracy_score(y_true, pred)}


def bootstrap_auc(y_true, probas: dict, n=1000):
    rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    idx = rng.integers(0, len(y_true), (n, len(y_true)))
    aucs = {k: np.array([roc_auc_score(y_true[i], p[i]) for i in idx]) for k, p in probas.items()}
    return aucs


# --------------------------------------------------------------------------- #
# 5. Plots
# --------------------------------------------------------------------------- #
def plot_strategy_comparison(comp: pd.DataFrame) -> None:
    r = comp.reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    sns.barplot(data=r, x="Model", y="roc_auc", hue="Strategy", ax=ax[0], palette="Set2")
    ax[0].set(title="5-fold CV ROC-AUC by imbalance strategy", ylim=(0.78, 0.87), ylabel="ROC-AUC")
    ax[0].axhline(0.80, c="r", ls="--", lw=1, label="0.80 target")
    ax[0].legend(fontsize=8)
    sns.barplot(data=r, x="Model", y="recall", hue="Strategy", ax=ax[1], palette="Set2")
    ax[1].set(title="Churner recall at the default 0.5 threshold", ylabel="recall")
    ax[1].get_legend().remove()
    plt.tight_layout()
    plt.savefig(FIG / "02_strategy_comparison.png", dpi=130)
    plt.close()


def plot_roc_pr(y_test, probas: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_test, p)
        ax[1].plot(rc, pr, lw=2, label=f"{name} (AP = {average_precision_score(y_test, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax[0].set(title="ROC curves - held-out test set", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend(loc="lower right")
    ax[1].axhline(y_test.mean(), c="k", ls="--", lw=1, label=f"no-skill ({y_test.mean():.2f})")
    ax[1].set(title="Precision-Recall curves", xlabel="Recall", ylabel="Precision")
    ax[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG / "03_roc_pr_curves.png", dpi=130)
    plt.close()


def plot_confusion(y_test, probas: dict, thresholds: dict) -> None:
    fig, ax = plt.subplots(1, len(probas), figsize=(5.2 * len(probas), 4.6))
    for a, (name, p) in zip(np.atleast_1d(ax), probas.items()):
        pred = (p >= thresholds[name]).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=a, colorbar=False)
        a.set_title(f"{name}\nthreshold = {thresholds[name]:.2f}")
    plt.tight_layout()
    plt.savefig(FIG / "04_confusion_matrices.png", dpi=130)
    plt.close()


def top_features(pipe: ImbPipeline, kind: str) -> pd.Series:
    names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    imp = clf.feature_importances_
    return pd.Series(imp, index=names).sort_values(ascending=False)


def plot_feature_importance(tuned: dict) -> pd.DataFrame:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    tidy = []
    for a, (name, pipe) in zip(ax, tuned.items()):
        s = top_features(pipe, name).head(10)
        tidy += [{"model": name, "rank": i + 1, "feature": f, "importance": v} for i, (f, v) in enumerate(s.items())]
        s.sort_values().plot.barh(ax=a, color="#4c72b0" if name == "Random Forest" else "#dd8452")
        a.set(title=f"{name} - top 10 features ({'impurity decrease' if name == 'Random Forest' else 'gain'})", xlabel="importance")
    plt.tight_layout()
    plt.savefig(FIG / "05_top10_feature_importance.png", dpi=130)
    plt.close()
    return pd.DataFrame(tidy)


def plot_permutation(tuned: dict, X_test, y_test) -> pd.DataFrame:
    """Permutation importance on ORIGINAL columns, measured as drop in test ROC-AUC (more trustworthy than
    impurity importance, which is biased towards high-cardinality / continuous features)."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    rows = {}
    for a, (name, pipe) in zip(ax, tuned.items()):
        r = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=1)
        s = pd.Series(r.importances_mean, index=X_test.columns).sort_values(ascending=False)
        err = pd.Series(r.importances_std, index=X_test.columns)[s.index]
        rows[name] = s
        top = s.head(10)
        a.barh(top.index[::-1], top.values[::-1], xerr=err[top.index][::-1].values, color="#55a868")
        a.set(title=f"{name} - permutation importance (drop in test AUC)", xlabel="mean decrease in ROC-AUC")
    plt.tight_layout()
    plt.savefig(FIG / "06_permutation_importance.png", dpi=130)
    plt.close()
    return pd.DataFrame(rows)


def plot_threshold(y_oof, proba_oof: dict) -> None:
    fig, ax = plt.subplots(1, len(proba_oof), figsize=(6 * len(proba_oof), 4.6), sharey=True)
    for a, (name, p) in zip(np.atleast_1d(ax), proba_oof.items()):
        ts = np.linspace(0.05, 0.9, 86)
        pr = [precision_score(y_oof, p >= t, zero_division=0) for t in ts]
        rc = [recall_score(y_oof, p >= t) for t in ts]
        f1 = [f1_score(y_oof, p >= t) for t in ts]
        a.plot(ts, pr, label="precision"); a.plot(ts, rc, label="recall"); a.plot(ts, f1, label="F1", lw=2.5)
        t_best = best_f1_threshold(y_oof, p)
        a.axvline(t_best, c="k", ls=":", label=f"best-F1 threshold = {t_best:.2f}")
        a.set(title=f"{name}: threshold trade-off (out-of-fold, train data)", xlabel="decision threshold")
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "07_threshold_tradeoff.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    df = clean(load_data(args.url))
    run_eda(df)

    X, y = df.drop(columns=TARGET), df[TARGET]
    cat_cols = [c for c in X.columns if c not in NUM]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"\nTrain {X_train.shape} | Test {X_test.shape} | scale_pos_weight = {spw:.3f}")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    print("\n=== STEP 1: imbalance strategy comparison (5-fold CV on train) ===")
    comp = compare_strategies(X_train, y_train, cat_cols, spw, cv)
    comp.round(4).to_csv(ROOT / "reports" / "strategy_comparison.csv")
    plot_strategy_comparison(comp)

    print("\n=== STEP 2: hyper-parameter tuning (RandomizedSearchCV, scoring = ROC-AUC) ===")
    n_iter = 6 if args.quick else 15
    tuned, searches = {}, {}
    for model in ("Random Forest", "XGBoost"):
        # strategy per model = best CV ROC-AUC among the two real balancing methods
        # ("none" is only a control; differences between strategies are within CV noise)
        sub = comp.loc[model].drop(index="none")
        strat = sub["roc_auc"].idxmax()
        print(f"[select] {model}: best balancing method by CV AUC -> {strat}")
        searches[model] = tune(model, strat, X_train, y_train, cat_cols, spw, cv, n_iter)
        tuned[model] = searches[model].best_estimator_
    # baseline for context
    baseline = build("Logistic Regression", "class weights", cat_cols, spw).fit(X_train, y_train)

    print("\n=== STEP 3: threshold from out-of-fold predictions (no test leakage) ===")
    oof = {m: cross_val_predict(tuned[m], X_train, y_train, cv=cv, method="predict_proba")[:, 1] for m in tuned}
    thr = {m: best_f1_threshold(y_train, oof[m]) for m in tuned}
    plot_threshold(y_train, oof)
    print({m: round(t, 3) for m, t in thr.items()})

    print("\n=== STEP 4: held-out test evaluation ===")
    probas = {m: tuned[m].predict_proba(X_test)[:, 1] for m in tuned}
    probas["Logistic Regression (baseline)"] = baseline.predict_proba(X_test)[:, 1]
    thr["Logistic Regression (baseline)"] = best_f1_threshold(
        y_train, cross_val_predict(baseline, X_train, y_train, cv=cv, method="predict_proba")[:, 1])
    boots = bootstrap_auc(y_test, probas)
    rows = []
    for m, p in probas.items():
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        row = {"Model": m, "ROC-AUC": roc_auc_score(y_test, p), "AUC 95% CI low": lo, "AUC 95% CI high": hi,
               "PR-AUC": average_precision_score(y_test, p)}
        d5, dt = metrics_at(y_test, p, 0.5), metrics_at(y_test, p, thr[m])
        row.update({f"{k}@0.5": v for k, v in d5.items() if k != "threshold"})
        row.update({f"{k}@tuned": v for k, v in dt.items()})
        rows.append(row)
    results = pd.DataFrame(rows).set_index("Model").round(4)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(results.T.to_string())
    results.to_csv(ROOT / "reports" / "test_results.csv")

    diff = boots["XGBoost"] - boots["Random Forest"]
    print(f"\nPaired bootstrap AUC(XGB) - AUC(RF): mean={diff.mean():+.4f}, 95% CI [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]")
    ok = all(results.loc[m, "ROC-AUC"] > 0.80 for m in tuned)
    print(f"ROC-AUC > 0.80 target met by both ensembles on the test set: {ok}")

    print("\n=== STEP 5: curves and feature importance ===")
    plot_roc_pr(y_test, probas)
    plot_confusion(y_test, {m: probas[m] for m in tuned}, {m: thr[m] for m in tuned})
    imp = plot_feature_importance(tuned)
    print("Top-10 (model-native importance):\n", imp.pivot(index="rank", columns="model", values="feature").to_string())
    imp.to_csv(ROOT / "reports" / "top10_feature_importance.csv", index=False)
    perm = plot_permutation(tuned, X_test, y_test)
    perm = perm.loc[perm.mean(axis=1).sort_values(ascending=False).index]
    perm.round(4).to_csv(ROOT / "reports" / "permutation_importance.csv")
    print("\nTop-5 permutation importance (original columns):\n", perm.head(5).round(4).to_string())

    best = max(tuned, key=lambda m: results.loc[m, "ROC-AUC"])
    joblib.dump({"model": tuned[best], "threshold": thr[best]}, ROOT / "models" / "best_model.joblib")
    (ROOT / "models" / "metadata.json").write_text(json.dumps({
        "best_model": best, "threshold": thr[best], "test_roc_auc": float(results.loc[best, "ROC-AUC"]),
        "features": list(X.columns), "seed": SEED,
        "best_params": {m: {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in searches[m].best_params_.items()} for m in searches}}, indent=2))
    print(f"\nSaved best model: {best} -> models/best_model.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--quick", action="store_true", help="fewer search iterations")
    main(ap.parse_args())
