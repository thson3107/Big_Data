import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.loader import load_all_coins
from src.data.cleaner import clean_crypto_dataframe
from src.features.builder import build_features
from src.mining.association import mine_association_rules, compare_rules_by_regime
from src.mining.clustering import cluster_coin_profiles
from src.mining.anomaly import detect_market_anomalies
from src.models.supervised import train_trend_classifier
from src.models.forecasting import run_forecasting_suite
from src.evaluation.metrics import classification_metrics_table
from src.evaluation.report import export_summary_report, pdf_summary, save_table
from src.visualization.plots import setup_style, plot_top_rules, plot_cluster_profiles, plot_forecast, plot_residual_regime, plot_error_by_shock


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict):
    for key in ["processed_dir", "outputs_dir", "figures_dir", "tables_dir", "models_dir", "reports_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def run_pipeline(cfg: dict):
    ensure_dirs(cfg)
    raw_df = load_all_coins(cfg["paths"]["raw_dir"])
    clean_df = clean_crypto_dataframe(raw_df, min_rows_per_coin=cfg["preprocessing"]["min_rows_per_coin"])
    feat_df = build_features(
        clean_df,
        forecast_horizon=cfg["features"]["forecast_horizon"],
        trend_horizon=cfg["features"]["trend_horizon"],
        vol_window=cfg["features"]["vol_window"],
    )

    processed_csv = Path(cfg["paths"]["processed_dir"]) / "crypto_features.csv"
    processed_parquet = Path(cfg["paths"]["processed_dir"]) / "crypto_features.parquet"
    feat_df.to_csv(processed_csv, index=False)
    feat_df.to_parquet(processed_parquet, index=False)

    tables_dir = Path(cfg["paths"]["tables_dir"])
    raw_overview_df = pd.DataFrame([
        {
            "rows": int(len(raw_df)),
            "columns": int(raw_df.shape[1]),
            "coins": int(raw_df["coin"].nunique()),
            "min_date": str(raw_df["date"].min()),
            "max_date": str(raw_df["date"].max()),
        }
    ])
    raw_missing_df = raw_df.isna().sum().rename("missing_count").reset_index().rename(columns={"index": "column"})
    raw_missing_df["missing_ratio"] = (raw_missing_df["missing_count"] / len(raw_df)).round(6)
    raw_by_coin_df = (
        raw_df.groupby("coin")
        .agg(rows=("coin", "size"), min_date=("date", "min"), max_date=("date", "max"), close_mean=("close", "mean"), volume_mean=("volume", "mean"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    preprocess_trace_df = pd.DataFrame([
        {"step": "load_raw", "rows": int(len(raw_df)), "columns": int(raw_df.shape[1]), "description": "Doc tat ca file coin_*.csv"},
        {"step": "clean", "rows": int(len(clean_df)), "columns": int(clean_df.shape[1]), "description": "Noi suy missing numeric, loc du lieu bat thuong"},
        {"step": "feature", "rows": int(len(feat_df)), "columns": int(feat_df.shape[1]), "description": "Tao return, volatility, momentum, RSI, state bins"},
    ])
    vol_ts_df = (
        feat_df.groupby("date")
        .agg(avg_volatility_14=("volatility_14", "mean"), avg_abs_return=("return_1d", lambda s: float(np.mean(np.abs(s)))), up_ratio=("trend_up", "mean"))
        .reset_index()
        .sort_values("date")
    )

    rules_df = mine_association_rules(
        feat_df,
        min_support=cfg["association"]["min_support"],
        min_confidence=cfg["association"]["min_confidence"],
        min_lift=cfg["association"]["min_lift"],
        top_k=cfg["association"]["top_k"],
    )
    regime_rules_df = compare_rules_by_regime(
        feat_df,
        min_support=cfg["association"]["min_support"],
        min_confidence=cfg["association"]["min_confidence"],
        min_lift=cfg["association"]["min_lift"],
        top_k=10,
    )
    cluster_df, centroid_df = cluster_coin_profiles(
        feat_df,
        n_clusters=cfg["clustering"]["n_clusters"],
        random_state=cfg["clustering"]["random_state"],
    )
    anomaly_df = detect_market_anomalies(feat_df, contamination=0.05, random_state=cfg["seed"])

    cls_model = None
    cls_metrics = {}
    cls_pred_df = pd.DataFrame()
    error_by_shock_df = pd.DataFrame()
    cls_report_df = pd.DataFrame()
    if cfg["classification"]["enabled"]:
        cls_model, cls_metrics, cls_pred_df, error_by_shock_df, cls_report_df = train_trend_classifier(
            feat_df,
            model_name=cfg["classification"]["model"],
            params=cfg["classification"]["params"],
            test_size_ratio=cfg["classification"]["test_size_ratio"],
        )

    forecast_metrics_df, forecast_df, residual_regime_df = run_forecasting_suite(
        feat_df,
        test_horizon=cfg["forecasting"]["test_horizon"],
        arima_order=tuple(cfg["forecasting"]["arima_order"]),
        ets_trend=cfg["forecasting"]["ets_trend"],
        ets_seasonal=cfg["forecasting"]["ets_seasonal"],
    )

    save_table(raw_overview_df, str(tables_dir / "raw_overview.csv"))
    save_table(raw_missing_df, str(tables_dir / "raw_missing_by_column.csv"))
    save_table(raw_by_coin_df, str(tables_dir / "raw_by_coin.csv"))
    save_table(preprocess_trace_df, str(tables_dir / "preprocess_trace.csv"))
    save_table(vol_ts_df, str(tables_dir / "volatility_timeseries.csv"))
    save_table(rules_df, str(tables_dir / "top_association_rules.csv"))
    save_table(regime_rules_df, str(tables_dir / "rules_by_regime.csv"))
    save_table(cluster_df, str(tables_dir / "coin_clusters.csv"))
    save_table(centroid_df, str(tables_dir / "cluster_centroids.csv"))
    save_table(anomaly_df, str(tables_dir / "anomalies.csv"))
    save_table(forecast_metrics_df, str(tables_dir / "forecast_metrics.csv"))
    save_table(forecast_df, str(tables_dir / "forecast_predictions.csv"))
    save_table(residual_regime_df, str(tables_dir / "residual_by_regime.csv"))

    if cfg["classification"]["enabled"]:
        cls_metrics_df = classification_metrics_table(cls_metrics)
        confusion_df = pd.crosstab(cls_pred_df["target_up"], cls_pred_df["pred_up"]).reset_index()
        confusion_df.columns = ["actual", "pred_0", "pred_1"]
        cls_by_regime_df = (
            cls_pred_df.groupby("state_vol", observed=False)
            .agg(accuracy=("is_error", lambda s: float(1 - s.mean())), avg_prob_up=("pred_prob_up", "mean"), samples=("is_error", "size"))
            .reset_index()
        )
        save_table(cls_metrics_df, str(tables_dir / "classification_metrics.csv"))
        save_table(cls_pred_df, str(tables_dir / "classification_predictions.csv"))
        save_table(error_by_shock_df, str(tables_dir / "classification_error_shock.csv"))
        save_table(cls_report_df, str(tables_dir / "classification_report.csv"))
        save_table(confusion_df, str(tables_dir / "classification_confusion.csv"))
        save_table(cls_by_regime_df, str(tables_dir / "classification_by_regime.csv"))
        joblib.dump(cls_model, Path(cfg["paths"]["models_dir"]) / "trend_classifier.pkl")

    setup_style()
    figures_dir = Path(cfg["paths"]["figures_dir"])
    plot_top_rules(rules_df, str(figures_dir / "top_rules.png"))
    plot_cluster_profiles(cluster_df, str(figures_dir / "cluster_profiles.png"))
    plot_forecast(forecast_df, str(figures_dir / "forecast_comparison.png"))
    plot_residual_regime(residual_regime_df, str(figures_dir / "residual_regime.png"))
    if cfg["classification"]["enabled"]:
        plot_error_by_shock(error_by_shock_df, str(figures_dir / "classification_error_shock.png"))

    forecast_best = forecast_metrics_df.sort_values("rmse").iloc[0].to_dict()
    payload = {
        "rows_raw": int(len(raw_df)),
        "rows_clean": int(len(clean_df)),
        "rows_featured": int(len(feat_df)),
        "n_coins": int(feat_df["coin"].nunique()),
        "top_forecast_model": forecast_best,
        "classification_metrics": cls_metrics,
    }
    export_summary_report(str(Path(cfg["paths"]["reports_dir"]) / "summary.json"), payload)

    sections = {
        "Tong quan du lieu": f"So dong du lieu sau feature: {len(feat_df)} cho {feat_df['coin'].nunique()} coin",
        "Khai pha luat ket hop": f"So luat top: {len(rules_df)}",
        "Phan cum": f"So cum: {cluster_df['cluster'].nunique()}",
        "Forecast": f"Model tot nhat theo RMSE: {forecast_best['model']} ({forecast_best['rmse']:.4f})",
        "Classification": json.dumps(cls_metrics, ensure_ascii=False),
    }
    pdf_summary(str(Path(cfg["paths"]["reports_dir"]) / "final_report.pdf"), sections)
    return payload


def run_streamlit(cfg: dict):
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go

    st.set_page_config(page_title="Bảng điều khiển khai phá Crypto", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;700;800&display=swap');
        .stApp {background: linear-gradient(135deg, #EEF5DB 0%, #f7f7ef 40%, #EEF5DB 100%);} 
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {font-family: 'Be Vietnam Pro', sans-serif;}
        h1, h2, h3 {color: #4F6367;}
        .block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
        .metric-card {background: #ffffffaa; border: 1px solid #4F6367; border-radius: 14px; padding: 12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Đề 14. Dự báo giá/biến động crypto")
    st.markdown(
        "Link dataset Kaggle: [Cryptocurrency Historical Prices dataset](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory)"
    )


    if st.button("Chạy lại pipeline", type="primary"):
        with st.spinner("Đang chạy pipeline..."):
            summary = run_pipeline(cfg)
        st.success("Đã cập nhật kết quả")
        st.json(summary)

    tables_dir = Path(cfg["paths"]["tables_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])

    def _load_csv(name: str):
        path = tables_dir / name
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    raw_overview = _load_csv("raw_overview.csv")
    raw_missing = _load_csv("raw_missing_by_column.csv")
    raw_by_coin = _load_csv("raw_by_coin.csv")
    preprocess_trace = _load_csv("preprocess_trace.csv")
    vol_ts = _load_csv("volatility_timeseries.csv")
    rules = _load_csv("top_association_rules.csv")
    rules_regime = _load_csv("rules_by_regime.csv")
    clusters = _load_csv("coin_clusters.csv")
    centroids = _load_csv("cluster_centroids.csv")
    anomalies = _load_csv("anomalies.csv")
    fm = _load_csv("forecast_metrics.csv")
    fp = _load_csv("forecast_predictions.csv")
    residual_regime = _load_csv("residual_by_regime.csv")
    cm = _load_csv("classification_metrics.csv")
    cr = _load_csv("classification_report.csv")
    cshock = _load_csv("classification_error_shock.csv")
    cconf = _load_csv("classification_confusion.csv")
    cregime = _load_csv("classification_by_regime.csv")
    cpred = _load_csv("classification_predictions.csv")

    if (reports_dir / "summary.json").exists():
        with open(reports_dir / "summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Số dòng sau đặc trưng", summary.get("rows_featured", 0))
        c2.metric("Số coin", summary.get("n_coins", 0))
        top_model = summary.get("top_forecast_model", {}).get("model", "N/A")
        c3.metric("Mô hình dự báo tốt nhất", top_model)

    tab1, tab2, tab3, tab4 = st.tabs(["EDA dữ liệu gốc", "Tiền xử lý + Khai phá", "Mô hình phân lớp", "Dự báo chuỗi thời gian"])

    with tab1:
        st.subheader("Thông tin dữ liệu gốc")
        if not raw_overview.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Số dòng raw", int(raw_overview.loc[0, "rows"]))
            c2.metric("Số cột", int(raw_overview.loc[0, "columns"]))
            c3.metric("Số coin", int(raw_overview.loc[0, "coins"]))
            c4.metric("Khoảng thời gian", f"{raw_overview.loc[0, 'min_date'][:10]} -> {raw_overview.loc[0, 'max_date'][:10]}")
        if not raw_by_coin.empty:
            st.markdown("Dữ liệu theo coin")
            st.dataframe(raw_by_coin, width="stretch")
            top_coin = raw_by_coin.head(12).copy()
            fig_coin = px.bar(top_coin, x="coin", y="rows", color="coin", color_discrete_sequence=["#4F6367", "#FE5F55", "#4F6367", "#FE5F55", "#4F6367", "#FE5F55", "#4F6367", "#FE5F55", "#4F6367", "#FE5F55", "#4F6367", "#FE5F55"])
            fig_coin.update_traces(marker_line_color="#4F6367", marker_line_width=1)
            fig_coin.update_layout(plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367", showlegend=False)
            st.plotly_chart(fig_coin, width="stretch")
        if not raw_missing.empty:
            st.markdown("Thiếu dữ liệu theo cột")
            st.dataframe(raw_missing, width="stretch")
            fig_missing = px.scatter(raw_missing, x="column", y="missing_ratio", color="column", color_discrete_sequence=["#4F6367", "#FE5F55", "#EEF5DB", "#4F6367", "#FE5F55", "#EEF5DB", "#4F6367", "#FE5F55", "#EEF5DB", "#4F6367"])
            fig_missing.update_traces(marker=dict(size=12, line=dict(width=1, color="#4F6367")))
            fig_missing.update_layout(plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367", showlegend=False, yaxis_title="Tỷ lệ thiếu", xaxis_title="Cột", yaxis=dict(range=[-0.02, max(float(raw_missing["missing_ratio"].max()) * 1.15, 0.05)]))
            st.plotly_chart(fig_missing, width="stretch")
            if float(raw_missing["missing_ratio"].max()) == 0:
                st.info("Đề 14. Dự báo giá/biến động crypto")

    with tab2:
        st.subheader("Bạn đã xử lý những gì")
        if not preprocess_trace.empty:
            st.dataframe(preprocess_trace, width="stretch")
        if not vol_ts.empty:
            vol_ts["date"] = pd.to_datetime(vol_ts["date"])
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=vol_ts["date"], y=vol_ts["avg_volatility_14"], mode="lines", name="avg_volatility_14", line=dict(color="#FE5F55", width=2)))
            fig_vol.add_trace(go.Scatter(x=vol_ts["date"], y=vol_ts["avg_abs_return"], mode="lines", name="avg_abs_return", line=dict(color="#4F6367", width=2)))
            fig_vol.update_layout(title="Biến động thị trường theo thời gian", plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367")
            st.plotly_chart(fig_vol, width="stretch")
        st.subheader("Khai phá tri thức")
        if not rules.empty:
            st.markdown("Top association rules")
            st.dataframe(rules, width="stretch")
            fig_rules = px.bar(rules.head(12), x="lift", y="confidence", color="support", color_continuous_scale=["#EEF5DB", "#FE5F55"])
            fig_rules.update_layout(plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367")
            st.plotly_chart(fig_rules, width="stretch")
        if not rules_regime.empty:
            st.markdown("")
            # fig_reg = px.box(rules_regime, x="regime", y="lift", color="regime", color_discrete_sequence=["#4F6367", "#FE5F55", "#EEF5DB"])
            # fig_reg.update_layout(plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367", showlegend=False)
            # st.plotly_chart(fig_reg, width="stretch")
        if not clusters.empty:
            st.markdown("Profiling cụm coin")
            fig_cluster = px.scatter(clusters, x="avg_volatility_14", y="avg_return_1d", color="profile_label", hover_data=["coin", "cluster"], color_discrete_sequence=["#4F6367", "#FE5F55", "#EEF5DB", "#4F6367"])
            fig_cluster.update_layout(plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367")
            st.plotly_chart(fig_cluster, width="stretch")
            st.dataframe(centroids, width="stretch")
        if not anomalies.empty:
            anomaly_rate = float(anomalies["is_anomaly"].mean())
            st.metric("Tỷ lệ anomaly", f"{anomaly_rate:.2%}")

    with tab3:
        st.subheader("Kết quả mô hình phân lớp up/down")
        if not cm.empty:
            st.dataframe(cm, width="stretch")
            cm_dict = {row["metric"]: float(row["value"]) for _, row in cm.iterrows()}
            st.markdown(f"Nhận xét: ROC-AUC = {cm_dict.get('roc_auc', 0):.4f}, F1 = {cm_dict.get('f1', 0):.4f}, Accuracy = {cm_dict.get('accuracy', 0):.4f}.")
        if not cr.empty:
            st.markdown("Classification report")
            st.dataframe(cr, width="stretch")
        if not cconf.empty:
            fig_conf = px.bar(cconf.melt(id_vars=["actual"], var_name="pred", value_name="count"), x="actual", y="count", color="pred", barmode="group", color_discrete_sequence=["#4F6367", "#FE5F55"])
            fig_conf.update_layout(title="Confusion matrix dạng cột", plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367")
            st.plotly_chart(fig_conf, width="stretch")
        if not cregime.empty:
            st.markdown("Độ chính xác theo volatility regime")
            fig_creg = px.bar(cregime, x="state_vol", y="accuracy", color="state_vol", color_discrete_sequence=["#4F6367", "#FE5F55", "#EEF5DB"])
            fig_creg.update_layout(plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367", showlegend=False)
            st.plotly_chart(fig_creg, width="stretch")
        if not cshock.empty:
            shock_map = {0: "normal", 1: "shock"}
            cshock["shock_label"] = cshock["shock_event"].map(shock_map)
            fig_shock = px.scatter(cshock, x="shock_label", y="error_rate", color="shock_label", color_discrete_sequence=["#4F6367", "#FE5F55"])
            fig_shock.update_traces(marker=dict(size=14, line=dict(width=1, color="#4F6367")))
            fig_shock.update_layout(title="Lỗi mô hình khi thị trường sốc", plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367", showlegend=False, yaxis=dict(range=[-0.02, max(float(cshock["error_rate"].max()) * 1.15, 0.05)]), yaxis_title="Tỷ lệ lỗi", xaxis_title="Điều kiện thị trường")
            st.plotly_chart(fig_shock, width="stretch")
            if len(cshock) == 2:
                normal_err = float(cshock[cshock["shock_event"] == 0]["error_rate"].iloc[0])
                shock_err = float(cshock[cshock["shock_event"] == 1]["error_rate"].iloc[0])
                delta = shock_err - normal_err
                # st.markdown(f"Nhận xét: error khi shock {'tăng' if delta > 0 else 'giảm'} {abs(delta):.4f} so với ngày bình thường.")
                if normal_err == 0 and shock_err == 0:
                    st.info("")

    with tab4:
        st.subheader("So sánh mô hình dự báo chuỗi thời gian")
        if not fm.empty:
            st.dataframe(fm, width="stretch")
            fig_fm = px.bar(fm, x="model", y=["mae", "rmse"], barmode="group", color_discrete_sequence=["#4F6367", "#FE5F55"])
            fig_fm.update_layout(plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367")
            st.plotly_chart(fig_fm, width="stretch")
            best_row = fm.sort_values("rmse").iloc[0]
            worst_row = fm.sort_values("rmse", ascending=False).iloc[0]
            improve = (float(worst_row["rmse"]) - float(best_row["rmse"])) / max(float(worst_row["rmse"]), 1e-9)
            st.markdown(f"Nhận xét: model tốt nhất theo RMSE là {best_row['model']}, cải thiện {improve:.2%} so với model tệ nhất.")
        if not fp.empty:
            fp["date"] = pd.to_datetime(fp["date"])
            fp_long = fp.melt(id_vars=["date"], value_vars=["actual", "naive", "arima", "ets"], var_name="chuoi", value_name="gia")
            fig_line = px.line(
                fp_long,
                x="date",
                y="gia",
                color="chuoi",
                line_dash="chuoi",
                color_discrete_map={
                    "actual": "#4F6367",
                    "naive": "#FE5F55",
                    "arima": "#4F6367",
                    "ets": "#FE5F55",
                },
                line_dash_map={
                    "actual": "solid",
                    "naive": "solid",
                    "arima": "dash",
                    "ets": "dash",
                },
            )
            fig_line.update_layout(title="Dự báo theo thời gian", plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367")
            st.plotly_chart(fig_line, width="stretch")
        if not residual_regime.empty:
            rr = residual_regime.melt(id_vars=["state_vol"], var_name="model", value_name="rmse")
            rr = rr.dropna(subset=["rmse"]).copy()
            fig_rr = px.bar(
                rr,
                x="state_vol",
                y="rmse",
                color="model",
                pattern_shape="model",
                barmode="group",
                color_discrete_map={
                    "residual_naive": "#4F6367",
                    "residual_arima": "#FE5F55",
                    "residual_ets": "#4F6367",
                },
                pattern_shape_map={
                    "residual_naive": "",
                    "residual_arima": "",
                    "residual_ets": "/",
                },
            )
            fig_rr.update_layout(title="Sai số residual theo volatility regime", plot_bgcolor="#EEF5DB", paper_bgcolor="#EEF5DB", font_color="#4F6367")
            st.plotly_chart(fig_rr, width="stretch")
            missing_regime = residual_regime[residual_regime[["residual_naive", "residual_arima", "residual_ets"]].isna().any(axis=1)]["state_vol"].astype(str).tolist()
            if missing_regime:
                st.warning(f"")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/params.yaml")
    parser.add_argument("--mode", type=str, choices=["pipeline", "app"], default="pipeline")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.mode == "app":
        run_streamlit(cfg)
    else:
        summary = run_pipeline(cfg)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
