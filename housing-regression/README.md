# California Housing: OLS vs Ridge (L2) vs Lasso (L1)

End-to-end regression project: imputation, feature engineering (ratios + interactions), multicollinearity diagnostics (VIF),
leakage-safe preprocessing, `GridSearchCV` over regularisation strength, and a full RMSE / MAE / R² benchmark with diagnostics.

**Dataset:** [housing.csv](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)
(20,640 census blocks, 9 numeric features + `ocean_proximity`, target `median_house_value`)

## Deliverable
`notebooks/Task2_Housing_Ridge_Lasso_Regression.ipynb` is the executed regression notebook (metrics table + residual error plots). `src/` holds the same workflow as a runnable script.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/pipeline.py                 # ~2-3 min on one CPU core
python src/pipeline.py --drop-capped   # variant without the censored $500k rows
```
Outputs: `reports/figures/*.png`, `reports/benchmark_results.csv`, `reports/vif.csv`, `reports/coefficients_engineered.csv`, `models/best_model.joblib`.

## Structure
```
housing-regression/
├── data/housing.csv          # cached download
├── notebooks/                # executed .ipynb deliverable
├── models/                   # best_model.joblib + metadata.json
├── reports/                  # CSVs + figures/ (12 plots)
├── src/
│   ├── features.py           # engineer_features(), Winsorizer (importable -> model unpickles cleanly)
│   └── pipeline.py           # whole workflow
└── requirements.txt
```

## Workflow
| Step | What happens |
|---|---|
| Load + EDA | 207 nulls in `total_bedrooms`; target distribution, geographic map, correlation heatmap |
| Feature engineering | `log_rooms_per_household`, `log_population_per_household`, `bedrooms_per_room`; interactions `income × log_rooms`, `income × age` |
| Multicollinearity | VIF in three stages (below); raw counts dropped in favour of ratios; Ridge/Lasso as model-side remedy |
| Preprocessing (in-pipeline) | median impute → winsorise 0.5/99.5% → `StandardScaler`; `OneHotEncoder(drop="first")` for `ocean_proximity` |
| Split | 80/20, stratified on income bands |
| Models | OLS, Ridge, Lasso; `GridSearchCV` (5-fold, neg-RMSE) over alpha |
| Two feature sets | **Engineered** (13 features) and **Engineered + all pairwise interactions** (91 features) |
| Diagnostics | predicted-vs-actual, residual histogram + KDE, residual-vs-fitted, alpha curves, coefficients |

## Results (test set, seed 42)
Baseline "predict the mean": RMSE **$114,165**.

| Feature set | Model | RMSE | MAE | R² | CV RMSE | alpha |
|---|---|---|---|---|---|---|
| Engineered | OLS | 63,466 | 46,373 | 0.691 | 65,011 | - |
| Engineered | Ridge | 63,534 | 46,389 | 0.690 | 65,008 | 10 |
| Engineered | Lasso | 63,560 | 46,402 | 0.690 | 65,010 | 31.6 (1 of 13 zeroed) |
| + interactions | OLS | 58,892 | 41,999 | 0.734 | 59,399 | - |
| + interactions | Ridge | 58,874 | 41,976 | 0.734 | 59,385 | 0.32 |
| + interactions | Lasso | **58,846** | 42,022 | 0.734 | 59,347 | 3.16 (15 of 91 zeroed) |

### What the results actually say
- **Interactions help; regularisation barely does.** Adding interaction features cuts RMSE by about $4.6k (7%). Ridge, Lasso and OLS then differ by under $50 RMSE, far smaller than the fold-to-fold noise (CV std ≈ $300–800). With 16.5k training rows and ≤ 91 features, variance is not the bottleneck, so shrinkage has little to fix. Do not read "Lasso wins" into the last decimal.
- **Lasso's real benefit is sparsity.** It matches OLS accuracy while dropping 15 of 91 interaction features (`09_lasso_selected_interactions.png`).
- **The ceiling is the model, not the regulariser.** Residual-vs-fitted plots show structure; linear models can't capture the non-linear location effect. Gradient boosting or a random forest is the natural next step.
- **Censored target.** 965 rows (4.7%) are capped at $500,001, which creates the diagonal streak in the residual plots. `--drop-capped` removes them.

### Multicollinearity (VIF; > 10 is a red flag)
| Feature | Raw | Ratios replace counts | + interactions |
|---|---|---|---|
| households / total_bedrooms / total_rooms | 28.3 / 26.9 / 12.1 | removed | removed |
| median_income | 1.7 | 2.1 | **101** |
| income_x_log_rooms | - | - | **105** |
| latitude / longitude | 8.8 / 8.7 | 9.6 / 9.2 | 9.8 / 9.4 |

Replacing raw counts with ratios removes the severe collinearity. Product terms then reintroduce it, because a product correlates with its own parents; this is the expected cost of interactions, and the reason Ridge/Lasso are used. Latitude/longitude stay near 9-10 because they are geographically correlated. Centering features before multiplying would lower the interaction VIFs and is a good extension.

### Bug found and fixed during development (kept for transparency)
A first version used raw ratios. A few extreme blocks (up to 1,243 people per household) were multiplied into the interaction terms, giving one CV fold an RMSE of **$1.26M** and a test RMSE of $75k for Ridge. Log-transforming the ratios and adding a train-fitted `Winsorizer` fixed it, and grids that had hit their edges now have interior optima.

## Using the saved model
```python
import sys, joblib, pandas as pd
sys.path.insert(0, "src")                       # so Winsorizer can be unpickled
from features import engineer_features
model = joblib.load("models/best_model.joblib")
row = pd.DataFrame([dict(longitude=-122.23, latitude=37.88, housing_median_age=41,
    total_rooms=880, total_bedrooms=129, population=322, households=126,
    median_income=8.3252, ocean_proximity="NEAR BAY")])
print(model.predict(engineer_features(row)))   # ≈ $388k (actual block: $452.6k)
```

## Limitations
- 1990 census data; not a basis for real valuation.
- Linear models with a censored target; predictions above ~$500k are not meaningful.
- Single random split plus 5-fold CV; no repeated CV or significance test between models.

## Extension ideas
1. Compare with `HistGradientBoostingRegressor` / `RandomForestRegressor`; model `log(target)` with `TransformedTargetRegressor`.
2. Add `ElasticNet`, centered interactions, and geographic features (distance to coast / cities, cluster-similarity).
3. Repeated K-fold plus a paired test for model comparison; SHAP for interpretation.
4. Serve with FastAPI/Streamlit; add `pytest` and GitHub Actions CI.

## Links
- Dataset (raw CSV): https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
- Dataset folder / book repo: https://github.com/ageron/handson-ml2
- Linear models guide (OLS, Ridge, Lasso): https://scikit-learn.org/stable/modules/linear_model.html
- Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Lasso: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
- LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- ColumnTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
- SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- Regression metrics (RMSE, MAE, R²): https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
- Common pitfalls / data leakage: https://scikit-learn.org/stable/common_pitfalls.html
- Original paper (Pace & Barry, 1997): https://doi.org/10.1016/S0167-7152(96)00140-X
