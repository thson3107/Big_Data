import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame, forecast_horizon: int = 1, trend_horizon: int = 3, vol_window: int = 14) -> pd.DataFrame:
    out = df.copy().sort_values(["coin", "date"])
    g = out.groupby("coin", group_keys=False)

    out["return_1d"] = g["close"].pct_change()
    out["return_3d"] = g["close"].pct_change(3)
    out["return_7d"] = g["close"].pct_change(7)
    out["log_return_1d"] = np.log1p(out["return_1d"].fillna(0))
    out["hl_spread"] = (out["high"] - out["low"]) / out["open"]
    out["oc_spread"] = (out["close"] - out["open"]) / out["open"]

    for w in [7, 14, 30]:
        out[f"ma_{w}"] = g["close"].transform(lambda s: s.rolling(w).mean())
        out[f"std_{w}"] = g["close"].transform(lambda s: s.rolling(w).std())
        out[f"volatility_{w}"] = g["return_1d"].transform(lambda s: s.rolling(w).std())
        out[f"momentum_{w}"] = g["close"].transform(lambda s: s / s.shift(w) - 1)

    out["rsi_14"] = g["close"].transform(lambda s: _rsi(s, 14))
    out["future_return_1d"] = g["close"].shift(-forecast_horizon) / out["close"] - 1
    out["future_trend_up"] = (g["close"].shift(-trend_horizon) > out["close"]).astype(int)
    out["trend_up"] = (out["return_1d"] > 0).astype(int)
    out["vol_regime_value"] = g["return_1d"].transform(lambda s: s.rolling(vol_window).std())

    out["state_updown"] = pd.cut(
        out["return_1d"],
        bins=[-1.0, 0, 1.0],
        labels=["down", "up"],
        include_lowest=True,
    )
    out["state_vol"] = pd.cut(
        out["vol_regime_value"],
        bins=[0.0, 0.015, 0.04, np.inf],
        labels=["low_vol", "mid_vol", "high_vol"],
        include_lowest=True,
    )

    feature_cols = [
        "return_1d",
        "return_3d",
        "return_7d",
        "log_return_1d",
        "hl_spread",
        "oc_spread",
        "ma_7",
        "ma_14",
        "ma_30",
        "std_7",
        "std_14",
        "std_30",
        "volatility_7",
        "volatility_14",
        "volatility_30",
        "momentum_7",
        "momentum_14",
        "momentum_30",
        "rsi_14",
        "vol_regime_value",
    ]

    out[feature_cols] = out.groupby("coin")[feature_cols].transform(lambda x: x.replace([np.inf, -np.inf], np.nan))
    out = out.dropna(subset=feature_cols + ["future_return_1d", "future_trend_up", "trend_up", "state_updown", "state_vol"])
    return out
