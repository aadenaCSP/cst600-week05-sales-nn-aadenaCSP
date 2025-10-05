from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_report(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / np.maximum(1e-8, y_true))).mean() * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE%": mape}

def plot_pred_vs_actual(dates, y_true, y_pred, out_path="figures/pred_vs_actual.png", title="Predicted vs Actual"):
    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(dates, y_true, label="Actual")
    plt.plot(dates, y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_residual_hist(residuals, out_path="figures/residual_hist.png", title="Residuals Distribution"):
    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=30)
    plt.title(title)
    plt.xlabel("Error (y_true - y_pred)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
