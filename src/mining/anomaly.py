import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_market_anomalies(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    use_cols = ["return_1d", "volatility_14", "hl_spread", "volume", "marketcap"]
    sample = df.dropna(subset=use_cols).copy()
    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=300)
    sample["anomaly_score"] = model.fit_predict(sample[use_cols])
    sample["is_anomaly"] = (sample["anomaly_score"] == -1).astype(int)
    return sample[["date", "coin", "return_1d", "volatility_14", "is_anomaly"]]
