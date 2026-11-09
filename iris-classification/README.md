# Multi-Class Benchmark Classification Pipeline (Iris)

End-to-end supervised classification with **Scikit-Learn**: EDA, leakage-safe scaling, 80/20 stratified split,
**Logistic Regression vs Decision Tree**, cross-validated tuning, full metrics, confusion matrices and decision boundaries.

**Dataset:** [iris.csv](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) (150 rows, 4 features, 3 balanced classes)

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/pipeline.py             # full run with GridSearchCV tuning
python src/pipeline.py --no-tune   # baseline only
```
Outputs: `reports/figures/*.png`, `reports/benchmark_results.csv`, `models/best_model.joblib`.

## Project structure
```
iris-classification/
├── data/iris.csv              # cached download
├── models/                    # best_model.joblib + metadata.json
├── reports/
│   ├── benchmark_results.csv
│   └── figures/               # 01..07 PNG plots
├── src/pipeline.py            # whole pipeline
└── requirements.txt
```

## Pipeline steps
| # | Step | Implementation |
|---|------|----------------|
| 1 | Load data | `pd.read_csv(URL)` with local-cache / sklearn fallback |
| 2 | EDA | summary stats, class balance, Seaborn `pairplot`, correlation heatmap |
| 3 | Split | `train_test_split(test_size=0.2, stratify=y, random_state=42)` → 120 / 30 |
| 4 | Scale | `StandardScaler` **inside** a `Pipeline` (fit on train only → no leakage) |
| 5 | Train | `LogisticRegression`, `DecisionTreeClassifier` |
| 6 | Tune | `GridSearchCV`, stratified 5-fold, scoring = macro-F1 |
| 7 | Evaluate | accuracy, macro precision / recall / F1, per-class report, 5-fold CV accuracy |
| 8 | Visualise | confusion-matrix heatmaps, decision boundaries, tree plot, feature importance |
| 9 | Persist | `joblib` dump of best pipeline |

## Results (seed 42)
| Model | Accuracy | Precision | Recall | F1 | CV Acc (5-fold) |
|---|---|---|---|---|---|
| Logistic Regression (C=10) | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 ± 0.031 |
| Decision Tree (depth 3, gini) | 0.967 | 0.970 | 0.967 | 0.967 | 0.958 ± 0.026 |

> **Read this carefully:** the test set has only 30 samples, so one misclassification = 3.3 points.
> The gap between the models is not statistically meaningful; the cross-validation column is the more reliable comparison,
> and it shows both models performing about the same (~96%). Setosa is perfectly separable; all errors involve versicolor/virginica.

## Using the saved model
```python
import joblib, pandas as pd
model = joblib.load("models/best_model.joblib")
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
print(model.predict(sample))   # ['setosa']
```

## Design notes (what makes it "professional")
- **No data leakage:** scaling and tuning happen only on training data.
- **Stratified** split and CV keep class ratios stable.
- **Reproducible:** fixed seeds, pinned minimum versions.
- **Decision boundaries** require 2-D input, so they use petal length/width (the most discriminative features) with the *same tuned hyper-parameters*; the metrics above use all 4 features.
- A scaler is kept in the tree pipeline only for uniformity; trees are scale-invariant.

## Extension ideas
1. Switch to Wine: `sklearn.datasets.load_wine` and set `labels` accordingly.
2. Add SVM, KNN, Random Forest and compare with repeated stratified CV.
3. Add a ROC-AUC one-vs-rest plot and McNemar / paired t-test for model comparison.
4. Wrap the model in a FastAPI or Streamlit app.
5. Add `pytest` tests and a GitHub Actions workflow.

## Useful links
- Iris dataset (raw CSV): https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
- Seaborn data repo: https://github.com/mwaskom/seaborn-data
- Scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
- StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- Metrics (precision/recall/F1): https://scikit-learn.org/stable/modules/model_evaluation.html
- Pipelines & leakage: https://scikit-learn.org/stable/common_pitfalls.html
- GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- Seaborn pairplot: https://seaborn.pydata.org/generated/seaborn.pairplot.html
- Decision boundary display: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html
- Wine dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
