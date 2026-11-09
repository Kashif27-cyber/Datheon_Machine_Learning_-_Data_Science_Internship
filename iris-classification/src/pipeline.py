"""
Multi-Class Benchmark Classification Pipeline (Iris)
====================================================
End-to-end: load -> EDA -> split -> scale -> train -> tune -> evaluate -> visualise -> save.

Run:  python src/pipeline.py            (with hyper-parameter tuning)
      python src/pipeline.py --no-tune  (baseline only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load Iris with Pandas; cache locally; fall back to sklearn if offline."""
    cache = ROOT / "data" / "iris.csv"
    try:
        df = pd.read_csv(url)
        df.to_csv(cache, index=False)
    except Exception as exc:  # network failure
        print(f"[warn] could not download ({exc}); using cache/sklearn fallback")
        if cache.exists():
            df = pd.read_csv(cache)
        else:
            from sklearn.datasets import load_iris

            raw = load_iris(as_frame=True)
            df = raw.frame.rename(columns=lambda c: c.replace(" (cm)", "").replace(" ", "_"))
            df["species"] = df.pop("target").map(dict(enumerate(raw.target_names)))
    return df


# --------------------------------------------------------------------------- #
# 2. Exploratory Data Analysis
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame) -> None:
    print("\n=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print("\nClass balance:\n", df["species"].value_counts().to_string())
    print("\nSummary statistics:\n", df.describe().round(2).to_string())

    sns.set_theme(style="whitegrid", context="notebook")
    g = sns.pairplot(df, hue="species", diag_kind="kde", palette="viridis",
                     plot_kws={"alpha": 0.8, "s": 35})
    g.figure.suptitle("Iris - Pairplot by Species", y=1.02, fontsize=14)
    g.savefig(FIG / "01_pairplot.png", dpi=130, bbox_inches="tight")
    plt.close("all")

    plt.figure(figsize=(6, 5))
    sns.heatmap(df.drop(columns="species").corr(), annot=True, fmt=".2f",
                cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(FIG / "02_correlation.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# 3-4. Models
# --------------------------------------------------------------------------- #
def build_models() -> dict[str, Pipeline]:
    """Scaler lives INSIDE the pipeline -> fitted on train only (no data leakage)."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),  # trees don't need it; kept for a uniform pipeline
            ("clf", DecisionTreeClassifier(random_state=SEED)),
        ]),
    }


def tune(models: dict[str, Pipeline], X_train, y_train) -> dict[str, Pipeline]:
    """Hyper-parameter search with stratified 5-fold CV (training data only)."""
    grids = {
        "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        "Decision Tree": {"clf__max_depth": [2, 3, 4, 5, None],
                          "clf__min_samples_leaf": [1, 2, 4],
                          "clf__criterion": ["gini", "entropy"]},
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tuned = {}
    for name, pipe in models.items():
        gs = GridSearchCV(pipe, grids[name], cv=cv, scoring="f1_macro", n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"[tune] {name}: best={gs.best_params_}  cv_f1_macro={gs.best_score_:.4f}")
        tuned[name] = gs.best_estimator_
    return tuned


def evaluate(models, X_train, y_train, X_test, y_test, labels) -> pd.DataFrame:
    rows = []
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, model in models.items():
        pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (macro)": precision_score(y_test, pred, average="macro"),
            "Recall (macro)": recall_score(y_test, pred, average="macro"),
            "F1 (macro)": f1_score(y_test, pred, average="macro"),
            "CV Acc (mean)": cv_scores.mean(),
            "CV Acc (std)": cv_scores.std(),
        })
        print(f"\n--- {name}: classification report (test set) ---")
        print(classification_report(y_test, pred, target_names=labels, digits=3))
    return pd.DataFrame(rows).set_index("Model").round(4)


# --------------------------------------------------------------------------- #
# 5. Visualisations
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(models, X_test, y_test, labels) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.8))
    for ax, (name, model) in zip(np.atleast_1d(axes), models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test), labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels, linewidths=0.5)
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
    plt.suptitle("Confusion Matrices - Test Set", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG / "03_confusion_matrices.png", dpi=130)
    plt.close()


def plot_decision_boundaries(df, models, labels) -> None:
    """Boundaries need 2-D input -> refit each tuned model on petal length/width only."""
    feats = ["petal_length", "petal_width"]
    X = df[feats].values
    y = pd.Categorical(df["species"], categories=labels).codes
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    bg = ListedColormap(["#fde0dd", "#e0f3db", "#deebf7"])
    fg = ["#e41a1c", "#4daf4a", "#377eb8"]
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 400),
                         np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 400))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, fitted) in zip(axes, models.items()):
        from sklearn.base import clone
        pipe = clone(fitted).fit(Xtr, ytr)  # same tuned hyper-parameters, 2 features
        Z = pipe.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=bg, alpha=0.9)
        for k, lab in enumerate(labels):
            ax.scatter(Xtr[ytr == k, 0], Xtr[ytr == k, 1], c=fg[k], edgecolor="k", s=40, label=lab)
            ax.scatter(Xte[yte == k, 0], Xte[yte == k, 1], c=fg[k], edgecolor="k", s=160, marker="*")
        ax.set(title=f"{name}  (test acc = {pipe.score(Xte, yte):.3f})",
               xlabel="petal_length (cm)", ylabel="petal_width (cm)")
    axes[0].legend(loc="upper left", fontsize=8)
    plt.suptitle("Decision Boundaries (stars = test samples)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG / "04_decision_boundaries.png", dpi=130)
    plt.close()


def plot_model_insights(models, feature_names, labels) -> None:
    tree = models["Decision Tree"].named_steps["clf"]
    plt.figure(figsize=(14, 7))
    plot_tree(tree, feature_names=feature_names, class_names=labels,
              filled=True, rounded=True, fontsize=8)
    plt.title("Tuned Decision Tree (splits on standardised features)")
    plt.savefig(FIG / "05_decision_tree.png", dpi=130, bbox_inches="tight")
    plt.close()

    lr = models["Logistic Regression"].named_steps["clf"]
    imp = pd.DataFrame({"Decision Tree": tree.feature_importances_,
                        "LogReg |coef| (mean)": np.abs(lr.coef_).mean(axis=0)},
                       index=feature_names)
    imp = imp / imp.sum()
    imp.plot.barh(figsize=(7, 4), color=["#4c72b0", "#dd8452"])
    plt.title("Normalised Feature Importance")
    plt.tight_layout()
    plt.savefig(FIG / "06_feature_importance.png", dpi=130)
    plt.close()


def plot_benchmark(results: pd.DataFrame) -> None:
    cols = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    ax = results[cols].T.plot.bar(figsize=(8, 4.5), rot=0, ylim=(0.8, 1.02), colormap="Set2")
    ax.set_title("Model Benchmark - Test Set")
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIG / "07_benchmark.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    df = load_data(args.url)
    run_eda(df)

    feature_names = [c for c in df.columns if c != "species"]
    labels = sorted(df["species"].unique())
    X, y = df[feature_names], df["species"]

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

    models = build_models()
    for m in models.values():
        m.fit(X_train, y_train)
    print("\n##### BASELINE (default hyper-parameters) #####")
    baseline = evaluate(models, X_train, y_train, X_test, y_test, labels)
    print(baseline.to_string())

    if args.tune:
        models = tune(models, X_train, y_train)
        print("\n##### TUNED #####")
        results = evaluate(models, X_train, y_train, X_test, y_test, labels)
    else:
        results = baseline
    print("\n=== FINAL BENCHMARK ===\n", results.to_string())

    plot_confusion_matrices(models, X_test, y_test, labels)
    plot_decision_boundaries(df, models, labels)
    plot_model_insights(models, feature_names, labels)
    plot_benchmark(results)

    results.to_csv(ROOT / "reports" / "benchmark_results.csv")
    best = results["F1 (macro)"].idxmax()
    joblib.dump(models[best], ROOT / "models" / "best_model.joblib")
    (ROOT / "models" / "metadata.json").write_text(json.dumps(
        {"best_model": best, "labels": labels, "features": feature_names, "seed": SEED}, indent=2))
    print(f"\nSaved best model: {best}  ->  models/best_model.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DATA_URL)
    ap.add_argument("--no-tune", dest="tune", action="store_false", help="skip GridSearchCV")
    main(ap.parse_args())
