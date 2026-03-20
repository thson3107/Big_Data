import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _cluster_profile_label(row: pd.Series) -> str:
    if row["avg_return_1d"] > 0 and row["avg_volatility_14"] < 0.03:
        return "majors_stable"
    if row["avg_return_1d"] > 0 and row["avg_volatility_14"] >= 0.03:
        return "growth_high_beta"
    if row["avg_return_1d"] <= 0 and row["avg_volatility_14"] >= 0.04:
        return "high_risk_drawdown"
    return "defensive_mixed"


def cluster_coin_profiles(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    profile = (
        df.groupby("coin")
        .agg(
            avg_return_1d=("return_1d", "mean"),
            avg_return_7d=("return_7d", "mean"),
            avg_volatility_14=("volatility_14", "mean"),
            avg_volatility_30=("volatility_30", "mean"),
            avg_rsi=("rsi_14", "mean"),
            avg_hl_spread=("hl_spread", "mean"),
        )
        .reset_index()
    )
    features = profile.drop(columns=["coin"])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(x_scaled)
    profile["cluster"] = labels
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
    centroid_unscaled = pd.DataFrame(scaler.inverse_transform(centroids), columns=features.columns)
    centroid_unscaled["cluster"] = centroid_unscaled.index
    centroid_unscaled["profile_label"] = centroid_unscaled.apply(_cluster_profile_label, axis=1)
    profile = profile.merge(centroid_unscaled[["cluster", "profile_label"]], on="cluster", how="left")
    return profile, centroid_unscaled
