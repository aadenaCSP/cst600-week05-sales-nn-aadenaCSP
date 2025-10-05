import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error

from data_load import load_data
from features import make_features
from model_mlp import time_split_index, build_preprocessor, ridge_baseline, mlp_search, naive_predictions
from evaluate import regression_report, plot_pred_vs_actual, plot_residual_hist

def run(horizon:int=1):
    Path("outputs").mkdir(exist_ok=True, parents=True)
    # 1) Load data
    df = load_data()

    # 2) Features (leakage-safe)
    X, y, feature_cols, df_dates = make_features(df, horizon=horizon)

    # 3) Time-aware split
    n = len(X)
    idx_train, idx_val, idx_test = time_split_index(n, 0.7, 0.15)
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_train, y_val, y_test = y.iloc[idx_train], y.iloc[idx_val], y.iloc[idx_test]
    dates_test = df_dates["date"].iloc[idx_test]

    # 4) Preprocessing
    preproc, _, _ = build_preprocessor(feature_cols)

    # 5) Baselines
    # Naive t-1 (last observed) and t-7 (last week)
    naive_t1  = naive_predictions(df_dates, idx_test, season=None)
    naive_t7  = naive_predictions(df_dates, idx_test, season=7)
    # Ridge on engineered features
    ridge = ridge_baseline(preproc)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    # 6-7) MLP with time-aware CV
    gcv = mlp_search(preproc)
    gcv.fit(X_train, y_train)
    mlp_best = gcv.best_estimator_
    mlp_pred = mlp_best.predict(X_test)

    # 8) Evaluate
    reports = {
        "naive_t1": regression_report(y_test, naive_t1),
        "naive_t7": regression_report(y_test, naive_t7),
        "ridge":    regression_report(y_test, ridge_pred),
        "mlp":      regression_report(y_test, mlp_pred)
    }

    # Save charts
    plot_pred_vs_actual(dates_test, y_test, mlp_pred, "figures/pred_vs_actual.png",
                        title="MLP: Predicted vs Actual (Test)")
    plot_residual_hist((y_test - mlp_pred), "figures/residual_hist.png",
                       title="MLP: Residuals (Test)")

    # 9) Persist metrics + best params
    out = {
        "horizon": horizon,
        "cv_best_params": gcv.best_params_,
        "metrics": reports
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    # Console summary (copy into README/PPT)
    print("\n=== RESULTS (Test) ===")
    for k, v in reports.items():
        print(f"{k:>9}  MAE={v['MAE']:.3f}  RMSE={v['RMSE']:.3f}  R2={v['R2']:.3f}  MAPE%={v['MAPE%']:.2f}")
    print("\nBest MLP params:", gcv.best_params_)
    print("Saved: figures/pred_vs_actual.png, figures/residual_hist.png, outputs/metrics.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1, help="Forecast horizon in days (default: 1)")
    args = ap.parse_args()
    run(horizon=args.horizon)
