import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_model(name: str, params: dict):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            random_state=params.get("random_state", 42),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            n_jobs=-1,
        )
    if name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=params.get("random_state", 42))),
        ])
    return GradientBoostingClassifier(
        n_estimators=params.get("n_estimators", 250),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 3),
        random_state=params.get("random_state", 42),
    )


def train_trend_classifier(df: pd.DataFrame, model_name: str = "gradient_boosting", params: dict | None = None, test_size_ratio: float = 0.2):
    if params is None:
        params = {}
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
    target_col = "trend_up" if "trend_up" in df.columns else "future_trend_up"
    data = df.sort_values("date").dropna(subset=feature_cols + [target_col])
    split_idx = int(len(data) * (1 - test_size_ratio))
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]

    x_train = train[feature_cols]
    y_train = train[target_col].astype(int)
    x_test = test[feature_cols]
    y_test = test[target_col].astype(int)

    model = _build_model(model_name, params)
    t0 = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - t0

    proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(x_test)
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "train_seconds": float(train_time),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }

    result = test[["date", "coin", target_col, "state_vol", "return_1d"]].copy()
    result = result.rename(columns={target_col: "target_up"})
    result["pred_prob_up"] = proba
    result["pred_up"] = pred
    result["is_error"] = (result["pred_up"] != result["target_up"]).astype(int)

    shock_threshold = result["return_1d"].abs().quantile(0.98)
    result["shock_event"] = (result["return_1d"].abs() >= shock_threshold).astype(int)
    error_by_shock = result.groupby("shock_event")["is_error"].mean().reset_index(name="error_rate")

    class_report = classification_report(y_test, pred, output_dict=True)
    report_df = pd.DataFrame(class_report).T.reset_index().rename(columns={"index": "class"})

    return model, metrics, result, error_by_shock, report_df
