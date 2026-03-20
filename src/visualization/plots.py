from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

PALETTE = {
    "stormcloud": "#4F6367",
    "beige": "#EEF5DB",
    "sunset": "#FE5F55",
}


def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.facecolor"] = PALETTE["beige"]
    plt.rcParams["axes.facecolor"] = PALETTE["beige"]
    plt.rcParams["axes.edgecolor"] = PALETTE["stormcloud"]
    plt.rcParams["axes.labelcolor"] = PALETTE["stormcloud"]
    plt.rcParams["xtick.color"] = PALETTE["stormcloud"]
    plt.rcParams["ytick.color"] = PALETTE["stormcloud"]
    plt.rcParams["text.color"] = PALETTE["stormcloud"]


def _save(fig, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_rules(rules_df: pd.DataFrame, out_path: str):
    if rules_df.empty:
        return
    top = rules_df.head(10).copy()
    top["rule"] = top["antecedents"] + " -> " + top["consequents"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top, x="lift", y="rule", color=PALETTE["sunset"], ax=ax)
    ax.set_title("Top Association Rules by Lift")
    ax.set_xlabel("Lift")
    ax.set_ylabel("Rule")
    _save(fig, out_path)


def plot_cluster_profiles(profile_df: pd.DataFrame, out_path: str):
    if profile_df.empty:
        return
    labels = profile_df["profile_label"].dropna().unique().tolist()
    base_colors = [PALETTE["stormcloud"], PALETTE["sunset"], PALETTE["beige"]]
    palette = {label: base_colors[i % len(base_colors)] for i, label in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=profile_df,
        x="avg_volatility_14",
        y="avg_return_1d",
        hue="profile_label",
        style="cluster",
        palette=palette,
        s=130,
        ax=ax,
    )
    ax.set_title("Coin Clusters by Return and Volatility")
    ax.set_xlabel("Average Volatility (14d)")
    ax.set_ylabel("Average Return (1d)")
    _save(fig, out_path)


def plot_forecast(forecast_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(forecast_df["date"], forecast_df["actual"], color=PALETTE["stormcloud"], linewidth=2.5, label="Actual")
    ax.plot(forecast_df["date"], forecast_df["naive"], color=PALETTE["beige"], linewidth=1.8, label="Naive")
    ax.plot(forecast_df["date"], forecast_df["arima"], color=PALETTE["sunset"], linewidth=1.8, label="ARIMA")
    ax.plot(forecast_df["date"], forecast_df["ets"], color=PALETTE["stormcloud"], linewidth=1.3, linestyle="--", label="ETS")
    ax.set_title("Walk-forward Forecast Comparison (Bitcoin)")
    ax.legend()
    _save(fig, out_path)


def plot_residual_regime(residual_df: pd.DataFrame, out_path: str):
    long_df = residual_df.melt(id_vars=["state_vol"], var_name="model", value_name="rmse")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long_df, x="state_vol", y="rmse", hue="model", palette=[PALETTE["stormcloud"], PALETTE["sunset"], PALETTE["beige"]], ax=ax)
    ax.set_title("Residual RMSE by Volatility Regime")
    ax.set_xlabel("Regime")
    ax.set_ylabel("RMSE")
    _save(fig, out_path)


def plot_error_by_shock(error_df: pd.DataFrame, out_path: str):
    if error_df.empty:
        return
    temp = error_df.copy()
    temp["shock_event"] = temp["shock_event"].map({0: "normal", 1: "shock"})
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=temp, x="shock_event", y="error_rate", hue="shock_event", palette={"normal": PALETTE["stormcloud"], "shock": PALETTE["sunset"]}, legend=False, ax=ax)
    ax.set_title("Classification Error Rate During Shock vs Normal Days")
    ax.set_xlabel("Market Condition")
    ax.set_ylabel("Error Rate")
    _save(fig, out_path)
