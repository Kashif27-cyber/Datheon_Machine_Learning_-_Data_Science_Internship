"""Reusable feature code (kept in its own module so saved models unpickle cleanly)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

TARGET = "median_house_value"
CAT = ["ocean_proximity"]
# raw count columns are near-duplicates of each other (VIF >> 10) -> replaced by ratios
RAW_COUNTS = ["total_rooms", "total_bedrooms", "population", "households"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise transforms only (no statistics learnt) -> safe to apply before the split.
    NaNs propagate and are imputed later inside the Pipeline (train-only statistics).
    Heavy-tailed ratios are log1p-transformed so a handful of extreme blocks
    (e.g. 1,243 people per household) cannot dominate OLS / interaction terms."""
    out = df.copy()
    out["log_rooms_per_household"] = np.log1p(out["total_rooms"] / out["households"])
    out["log_population_per_household"] = np.log1p(out["population"] / out["households"])
    out["bedrooms_per_room"] = out["total_bedrooms"] / out["total_rooms"]
    # domain interaction features
    out["income_x_log_rooms"] = out["median_income"] * out["log_rooms_per_household"]
    out["income_x_age"] = out["median_income"] * out["housing_median_age"]
    return out


class Winsorizer(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learnt on the TRAINING data only."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)
