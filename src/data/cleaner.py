import numpy as np
import pandas as pd


def clean_crypto_dataframe(df: pd.DataFrame, min_rows_per_coin: int = 200) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["coin", "date"])
    keep = out.groupby("coin").size()
    keep = keep[keep >= min_rows_per_coin].index
    out = out[out["coin"].isin(keep)].copy()
    for col in ["high", "low", "open", "close", "volume", "marketcap"]:
        out[col] = out.groupby("coin")[col].transform(lambda s: s.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both"))
    out = out.dropna(subset=["high", "low", "open", "close", "volume", "marketcap"])
    out = out[(out["high"] >= out["low"]) & (out["open"] > 0) & (out["close"] > 0)]
    out["dollar_volume"] = out["close"] * out["volume"]
    return out
