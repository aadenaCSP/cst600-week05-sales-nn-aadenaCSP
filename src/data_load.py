from pathlib import Path
import numpy as np
import pandas as pd

def generate_synthetic_ecommerce(start="2023-01-01", end="2025-06-30", seed=42) -> pd.DataFrame:
    """Reproducible synthetic daily e-commerce series with price/promo/holiday covariates."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    # Components
    trend = np.linspace(100, 140, n)                         # slow upward trend
    dow = dates.dayofweek
    weekly = 10*np.sin(2*np.pi*(dow)/7) + 5*(dow >= 5)       # weekly seasonality + weekend lift
    price = 20 + np.cumsum(rng.normal(0, 0.05, size=n))      # random walk around 20
    promo = rng.binomial(1, 0.10, size=n)                    # ~10% promo days
    is_holiday = ((dates.day == 1) |                          # first of each month
                  ((dates.month == 11) & (dates.day == 24)) | # sample 'holiday' markers
                  ((dates.month == 12) & (dates.day == 25))).astype(int)
    noise = rng.normal(0, 5, size=n)

    # Sales is affected by price (negative elasticity), promos, holidays, seasonality, and noise
    sales = trend + weekly - 4.0*(price-20) + 15*promo + 8*is_holiday + noise
    sales = np.clip(sales, 0, None)

    return pd.DataFrame({
        "date": dates,
        "sales": sales.astype(float),
        "price": price.astype(float),
        "promo": promo.astype(int),
        "is_holiday": is_holiday.astype(int)
    })

def load_data() -> pd.DataFrame:
    """
    Load a CSV from data/raw/sales.csv if present; otherwise generate synthetic data.
    CSV must include: date, sales, [optional: price, promo, is_holiday]
    """
    csv_path = Path("data/raw/sales.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"])
        # Ensure required columns exist; create reasonable defaults if missing
        if "price" not in df: df["price"] = df["sales"].rolling(7, min_periods=1).mean() / 5
        if "promo" not in df: df["promo"] = 0
        if "is_holiday" not in df:
            df["is_holiday"] = ((df["date"].dt.day == 1) | (df["date"].dt.weekday >= 5)).astype(int)
        return df.sort_values("date").reset_index(drop=True)
    else:
        return generate_synthetic_ecommerce()
