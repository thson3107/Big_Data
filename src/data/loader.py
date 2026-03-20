from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "SNo",
    "Name",
    "Symbol",
    "Date",
    "High",
    "Low",
    "Open",
    "Close",
    "Volume",
    "Marketcap",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.strip().lower() for c in df.columns}
    out = df.rename(columns=mapping)
    return out


def _read_coin_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if not set(REQUIRED_COLUMNS).issubset(set(df.columns)):
        alt = pd.read_csv(path, sep=",", header=0)
        if set(REQUIRED_COLUMNS).issubset(set(alt.columns)):
            df = alt
    df = _normalize_columns(df)
    numeric_cols = ["high", "low", "open", "close", "volume", "marketcap"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    coin_name = path.stem.replace("coin_", "")
    df["coin"] = coin_name
    return df


def load_all_coins(raw_dir: str) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob("coin_*.csv"))
    if not files:
        raise FileNotFoundError(f"Khong tim thay file csv tai {raw_dir}")
    frames = [_read_coin_file(f) for f in files]
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["coin", "date"])
    return data
