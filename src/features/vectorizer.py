"""TF-IDF feature engineering."""

from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def fit_vectorizer(
    texts: list[str],
    config: dict[str, Any] | None = None,
) -> TfidfVectorizer:
    cfg = (config or {}).get("features", {})
    ngrams = cfg.get("ngram_range", [1, 2])
    vectorizer = build_vectorizer(
        max_features=cfg.get("max_features", 10000),
        ngram_range=(ngrams[0], ngrams[1]),
        min_df=cfg.get("min_df", 2),
        max_df=cfg.get("max_df", 0.95),
    )
    vectorizer.fit(texts)
    return vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: Path) -> TfidfVectorizer:
    return joblib.load(path)
