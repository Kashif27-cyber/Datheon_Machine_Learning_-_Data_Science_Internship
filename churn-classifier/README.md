# Predictive Customer Churn Classifier (Random Forest & XGBoost)

Predict telecom customer churn under class imbalance, comparing **no balancing vs class weights (`scale_pos_weight`) vs SMOTE**,
tuning **Random Forest** and **XGBoost** for **ROC-AUC**, picking an honest decision threshold, and explaining the drivers of churn.

**Dataset:** [Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
(7,043 customers, 19 features, target `Churn`; 26.5% churn, i.e. a 2.8 : 1 imbalance)

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/churn_pipeline.py           # full run, ~4-5 min on one CPU core
python src/churn_pipeline.py --quick   # fewer search iterations, ~2 min
```
Outputs: `reports/figures/*.png`, `reports/*.csv`, `models/best_model.joblib`, `models/metadata.json`.

## Structure
```
churn-classifier/
├── data/telco_churn.csv        # cached download
├── models/                     # best_model.joblib (pipeline + threshold), metadata.json
├── reports/                    # CSV tables + figures/ (7 plots)
├── src/churn_pipeline.py       # whole workflow
└── requirements.txt
```

## Workflow
| Step | Implementation |
|---|---|
| Load | `pd.read_csv(URL)` with local-cache fallback |
| Clean | `TotalCharges` text → numeric. The 11 blank values all have `tenure == 0` (new, unbilled customers) so they are set to **0**, not a median (asserted in code). `SeniorCitizen` 0/1 → categorical; `customerID` dropped |
| EDA | class balance, churn by contract / internet service, tenure distribution |
| Split | 80/20 stratified (5,634 train / 1,409 test) |
| Encode | `OneHotEncoder(handle_unknown="ignore")` for 16 categorical columns; `StandardScaler` for 3 numeric |
| Imbalance | compare **none / class weights / SMOTE**. SMOTE lives *inside* an `imblearn` Pipeline, so it only touches training folds, never validation or test data |
| Models | `RandomForestClassifier`, `XGBClassifier` (+ Logistic Regression as a baseline) |
| Tuning | `RandomizedSearchCV`, 5-fold stratified CV, scoring = ROC-AUC |
| Threshold | best-F1 threshold chosen on **out-of-fold training predictions** (no test leakage) |
| Evaluation | test ROC-AUC with bootstrap 95% CI, PR-AUC, precision / recall / F1 / balanced accuracy at 0.5 and at the tuned threshold |
| Explainability | top-10 native importances + permutation importance (drop in test AUC) on the original columns |

## Results (held-out test set, seed 42)
**ROC-AUC target (> 0.80): met by both ensembles.**

| Model | ROC-AUC (95% CI) | PR-AUC | Precision | Recall | F1 | threshold |
|---|---|---|---|---|---|---|
| Random Forest (class weights) | 0.846 (0.823-0.866) | 0.658 | 0.556 | 0.749 | 0.638 | 0.55 |
| XGBoost (class weights) | **0.848** (0.826-0.869) | 0.666 | 0.562 | 0.727 | 0.634 | 0.60 |
| Logistic Regression (baseline) | 0.842 (0.819-0.862) | 0.633 | 0.539 | 0.725 | 0.618 | 0.58 |

### What the results actually say
- **The ensembles do not meaningfully beat a plain logistic regression.** Paired-bootstrap AUC difference XGBoost − RF = +0.002 (95% CI −0.004 to +0.007), which is indistinguishable from zero, and the 0.842 baseline sits inside both ensembles' confidence intervals. This dataset is mostly linearly separable; the extra model complexity buys little. Don't present "XGBoost wins" as a finding.
- **Balancing barely changes ROC-AUC, by design.** AUC depends only on ranking. In CV, "none" vs "class weights" differ by < 0.001 AUC and SMOTE is marginally *worse* for Random Forest (0.836 vs 0.841). What balancing really changes is **recall at the default 0.5 threshold**: XGBoost churner recall goes from 0.52 (unweighted) to 0.80 (weighted). If you only report AUC you miss the point of handling imbalance.
- **Threshold matters more than the sampler.** `reports/figures/07_threshold_tradeoff.png` shows the precision/recall trade-off; pick the operating point from retention-campaign costs, not from F1 by default.
- **"Severe" imbalance is an overstatement here.** 26.5% churn (2.8 : 1) is moderate; the techniques matter more on 1-in-100 problems.

### Strategy comparison (5-fold CV on train)
| Model | Strategy | ROC-AUC | Recall@0.5 | Precision@0.5 |
|---|---|---|---|---|
| Random Forest | none | 0.841 | 0.496 | 0.665 |
| Random Forest | class weights | 0.841 | 0.689 | 0.578 |
| Random Forest | SMOTE | 0.836 | 0.646 | 0.589 |
| XGBoost | none | 0.847 | 0.524 | 0.666 |
| XGBoost | class weights | 0.847 | 0.797 | 0.524 |
| XGBoost | SMOTE | 0.845 | 0.656 | 0.602 |

CV standard deviations are about 0.01, so AUC gaps between strategies are noise. The tuned strategy was chosen between the two real balancing methods (class weights vs SMOTE); "none" is kept as a control.

### Top churn drivers
Both models and the permutation analysis agree: **month-to-month contract** (by far the strongest), **short tenure**, **fiber-optic internet**, **no online security / no tech support**, and **electronic-check payment**.
Permutation importance (drop in test AUC): Contract 0.11 (RF) / 0.06 (XGB), tenure 0.03 / 0.05, InternetService 0.025 / 0.012.
These are **associations, not causes**: e.g. fiber customers may churn because of price or competition, not the fiber technology itself.

## Using the saved model
```python
import joblib, pandas as pd
bundle = joblib.load("models/best_model.joblib")      # {"model": pipeline, "threshold": float}
row = pd.DataFrame([dict(gender="Female", SeniorCitizen="No", Partner="Yes", Dependents="No", tenure=2,
    PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
    DeviceProtection="No", TechSupport="No", StreamingTV="Yes", StreamingMovies="Yes", Contract="Month-to-month",
    PaperlessBilling="Yes", PaymentMethod="Electronic check", MonthlyCharges=95.5, TotalCharges=190.0)])
p = bundle["model"].predict_proba(row)[0, 1]
print(p, p >= bundle["threshold"])                    # churn probability and flag
```

## Limitations
- One public dataset, one random split; results will move by a few thousandths with another seed.
- Features are static snapshots (no usage trends, complaints or price changes), so lift over the linear baseline is limited.
- Probabilities are not calibrated; calibrate (`CalibratedClassifierCV`) before using them as true risk estimates.
- SMOTE is applied after one-hot encoding, so synthetic points can have fractional dummy values; `SMOTENC` is an alternative.

## Extension ideas
1. Probability calibration + a cost-based threshold (cost of a retention offer vs value of a saved customer); uplift modelling.
2. Add LightGBM / CatBoost and SHAP explanations; repeated CV with paired tests.
3. Serve with FastAPI or Streamlit; add `pytest` tests and GitHub Actions CI.

## Links
- Dataset (raw CSV): https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Dataset repo: https://github.com/IBM/telco-customer-churn-on-icp4d
- RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- XGBoost docs: https://xgboost.readthedocs.io/en/stable/
- imbalanced-learn SMOTE: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
- ROC-AUC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- RandomizedSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
- OneHotEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- SMOTE paper (Chawla et al., 2002): https://doi.org/10.1613/jair.953
- XGBoost paper (Chen & Guestrin, 2016): https://doi.org/10.1145/2939672.2939785
