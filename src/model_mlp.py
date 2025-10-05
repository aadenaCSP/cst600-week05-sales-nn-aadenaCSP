from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def time_split_index(n:int, train_frac=0.7, val_frac=0.15):
    train_end = int(n*train_frac)
    val_end = int(n*(train_frac+val_frac))
    idx_train = np.arange(0, train_end)
    idx_val   = np.arange(train_end, val_end)
    idx_test  = np.arange(val_end, n)
    return idx_train, idx_val, idx_test

def build_preprocessor(feature_cols: List[str]) -> Tuple[ColumnTransformer, list[str], list[str]]:
    """Leakage-safe preprocessing (fit on train only via Pipeline)."""
    categorical = ["day_of_week","month","is_weekend","is_holiday","promo"]
    numeric = [c for c in feature_cols if c not in categorical]
    preproc = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ]), categorical)
    ])
    return preproc, numeric, categorical

def ridge_baseline(preproc: ColumnTransformer) -> Pipeline:
    return Pipeline([("preprocess", preproc), ("est", Ridge(alpha=1.0, random_state=42))])

def mlp_search(preproc: ColumnTransformer) -> GridSearchCV:
    pipe = Pipeline([("preprocess", preproc),
                     ("est", MLPRegressor(random_state=42, max_iter=400, shuffle=False))])
    # modest grid; time-aware CV
    param_grid = {
        "est__hidden_layer_sizes": [(32,), (64,), (64,32)],
        "est__alpha": [1e-4, 1e-3, 1e-2],
        "est__learning_rate_init": [1e-3, 5e-3],
        "est__activation": ["relu","tanh"]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    return GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=None, verbose=0)

def naive_predictions(df_dates: pd.DataFrame, idx_test: np.ndarray, season: int|None=None) -> np.ndarray:
    """
    Naïve baseline: predict last observed (season=None => t-1), or last week (season=7).
    Uses the actual sales at prior time steps; first test point uses last pre-test point.
    """
    y_all = df_dates["sales"].values
    y_pred = []
    for i in idx_test:
        lag = season if season else 1
        y_pred.append(y_all[i - lag])
    return np.array(y_pred)
