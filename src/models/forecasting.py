import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _walk_forward_naive(series: pd.Series, test_horizon: int):
    train = series.iloc[:-test_horizon]
    test = series.iloc[-test_horizon:]
    preds = []
    history = train.tolist()
    for actual in test:
        preds.append(history[-1])
        history.append(actual)
    return test.values, np.array(preds)


def _walk_forward_arima(series: pd.Series, test_horizon: int, order=(2, 1, 2)):
    train = series.iloc[:-test_horizon]
    test = series.iloc[-test_horizon:]
    preds = []
    history = train.tolist()
    for actual in test:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = ARIMA(history, order=order)
            fit = model.fit()
            pred = fit.forecast(steps=1)[0]
        preds.append(pred)
        history.append(actual)
    return test.values, np.array(preds)


def _walk_forward_ets(series: pd.Series, test_horizon: int, trend="add", seasonal=None):
    train = series.iloc[:-test_horizon]
    test = series.iloc[-test_horizon:]
    preds = []
    history = train.tolist()
    for actual in test:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(pd.Series(history), trend=trend, seasonal=seasonal)
            fit = model.fit(optimized=True)
            pred = fit.forecast(1).iloc[0]
        preds.append(pred)
        history.append(actual)
    return test.values, np.array(preds)


def _metric_row(model_name: str, y_true, y_pred):
    return {
        "model": model_name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
    }


def run_forecasting_suite(df: pd.DataFrame, test_horizon: int = 60, arima_order=(2, 1, 2), ets_trend="add", ets_seasonal=None):
    btc = df[df["coin"].str.lower().str.contains("bitcoin")].sort_values("date").copy()
    if len(btc) <= test_horizon + 100:
        raise ValueError("Khong du du lieu de forecast")
    series = btc["close"].astype(float).reset_index(drop=True)

    y_true_n, y_pred_n = _walk_forward_naive(series, test_horizon)
    y_true_a, y_pred_a = _walk_forward_arima(series, test_horizon, tuple(arima_order))
    y_true_e, y_pred_e = _walk_forward_ets(series, test_horizon, ets_trend, ets_seasonal)

    metrics = pd.DataFrame([
        _metric_row("Naive", y_true_n, y_pred_n),
        _metric_row("ARIMA", y_true_a, y_pred_a),
        _metric_row("ETS", y_true_e, y_pred_e),
    ]).sort_values("rmse")

    forecast_df = pd.DataFrame({
        "date": btc["date"].iloc[-test_horizon:].values,
        "actual": y_true_n,
        "naive": y_pred_n,
        "arima": y_pred_a,
        "ets": y_pred_e,
    })
    forecast_df["residual_naive"] = forecast_df["actual"] - forecast_df["naive"]
    forecast_df["residual_arima"] = forecast_df["actual"] - forecast_df["arima"]
    forecast_df["residual_ets"] = forecast_df["actual"] - forecast_df["ets"]

    vol = btc[["date", "state_vol"]].iloc[-test_horizon:].reset_index(drop=True)
    residual_by_regime = forecast_df.merge(vol, on="date", how="left")
    residual_by_regime = residual_by_regime.groupby("state_vol", observed=False)[["residual_naive", "residual_arima", "residual_ets"]].agg(lambda s: float(np.sqrt(np.mean(np.square(s))))).reset_index()

    return metrics, forecast_df, residual_by_regime
