import pandas as pd
from typing import Tuple, List

def make_features(df: pd.DataFrame, horizon:int=1) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    """
    Build calendar + lag + rolling features.
    IMPORTANT: All features use only past info (shifted where needed) to avoid leakage.
    Returns: X, y, feature_names, df_model_dates (date & sales aligned with X/y)
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lags (shift by 1+ to use PAST values only)
    df["sales_lag1"]  = df["sales"].shift(1)
    df["sales_lag7"]  = df["sales"].shift(7)
    df["price_lag1"]  = df["price"].shift(1)
    df["promo_lag1"]  = df["promo"].shift(1)

    # Rolling stats (on shifted sales so "today" is not in the window)
    df["sales_roll_mean_7"]  = df["sales"].shift(1).rolling(7).mean()
    df["sales_roll_std_7"]   = df["sales"].shift(1).rolling(7).std()
    df["sales_roll_mean_28"] = df["sales"].shift(1).rolling(28).mean()

    # Price momentum
    df["price_change_7"] = df["price"].pct_change(7)

    # Forecast target horizon
    df["target"] = df["sales"].shift(-horizon)

    # Drop rows with NaNs introduced by shifts/rollings and future shift
    df_model = df.dropna().reset_index(drop=True)

    feature_cols = [
        "price","promo","is_holiday","day_of_week","month","is_weekend",
        "sales_lag1","sales_lag7","price_lag1","promo_lag1",
        "sales_roll_mean_7","sales_roll_std_7","sales_roll_mean_28","price_change_7"
    ]
    X = df_model[feature_cols].copy()
    y = df_model["target"].copy()
    return X, y, feature_cols, df_model[["date","sales"]]
