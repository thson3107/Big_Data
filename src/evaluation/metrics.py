import pandas as pd


def classification_metrics_table(metrics: dict) -> pd.DataFrame:
    rows = [
        {"metric": "accuracy", "value": metrics.get("accuracy", 0.0)},
        {"metric": "f1", "value": metrics.get("f1", 0.0)},
        {"metric": "roc_auc", "value": metrics.get("roc_auc", 0.0)},
        {"metric": "train_seconds", "value": metrics.get("train_seconds", 0.0)},
        {"metric": "n_train", "value": metrics.get("n_train", 0)},
        {"metric": "n_test", "value": metrics.get("n_test", 0)},
    ]
    return pd.DataFrame(rows)
