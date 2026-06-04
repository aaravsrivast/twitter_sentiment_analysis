"""Load and prepare sentiment datasets (Sentiment140-style or CSV)."""

from pathlib import Path

import pandas as pd


def normalize_labels(series: pd.Series) -> pd.Series:
    """Map Sentiment140 labels (0=neg, 4=pos) and variants to binary 0/1."""
    mapped = series.copy()
    mapped = mapped.replace({4: 1, "4": 1, "positive": 1, "pos": 1, 1: 1})
    mapped = mapped.replace({0: 0, "0": 0, "negative": 0, "neg": 0})
    mapped = mapped.replace({2: 0, "2": 0, "neutral": 0})
    return mapped.astype(int)


def load_dataset(
    path: Path,
    text_col: str = "text",
    label_col: str = "sentiment",
) -> pd.DataFrame:
    """
    Load labeled tweets from CSV.

    Supports:
    - Standard CSV with columns: text, sentiment (0/1 or 0/4)
    - Sentiment140 encoding.csv (no header): target, id, date, flag, user, text
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
        if "text" not in df.columns and len(df.columns) >= 6:
            df = pd.read_csv(
                path,
                encoding="latin-1",
                header=None,
                names=["target", "id", "date", "flag", "user", "text"],
            )
            text_col, label_col = "text", "target"
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}'. Columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'. Columns: {list(df.columns)}")

    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "sentiment"]
    out = out.dropna(subset=["text", "sentiment"])
    out["sentiment"] = normalize_labels(out["sentiment"])
    out = out[out["sentiment"].isin([0, 1])]
    return out.reset_index(drop=True)
