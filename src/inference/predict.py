"""Real-time sentiment prediction on tweet text."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, get_path
from src.features.vectorizer import load_vectorizer
from src.models.train import load_classifier, _build_cleaner
from src.preprocessing.text_cleaner import TextCleaner

LABEL_MAP = {0: "negative", 1: "positive"}


class SentimentPredictor:
    def __init__(
        self,
        model_name: str = "logistic_regression",
        models_dir: Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or load_config()
        self.model_name = model_name
        models_dir = models_dir or get_path("models_dir", self.config)
        prefix = Path(models_dir) / model_name

        self.classifier = load_classifier(models_dir, model_name)
        self.vectorizer = load_vectorizer(Path(f"{prefix}_vectorizer.joblib"))
        self.cleaner = _build_cleaner(self.config)

    def predict_one(self, text: str) -> dict[str, Any]:
        cleaned = self.cleaner.clean(text)
        X = self.vectorizer.transform([cleaned])
        label = int(self.classifier.predict(X)[0])
        proba = None
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)[0]
            proba = {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}
        elif hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(X)[0]
            if np.ndim(scores) == 0:
                scores = [scores]
            exp = np.exp(scores - np.max(scores))
            norm = exp / exp.sum()
            proba = {LABEL_MAP[i]: float(p) for i, p in enumerate(norm)}

        return {
            "text": text,
            "cleaned_text": cleaned,
            "sentiment": LABEL_MAP[label],
            "label": label,
            "confidence": proba,
        }

    def predict_batch(self, texts: list[str]) -> pd.DataFrame:
        cleaned = self.cleaner.transform_iterable(texts)
        X = self.vectorizer.transform(cleaned)
        labels = self.classifier.predict(X)
        rows = []
        for raw, clean, lab in zip(texts, cleaned, labels):
            rows.append(
                {
                    "text": raw,
                    "cleaned_text": clean,
                    "sentiment": LABEL_MAP[int(lab)],
                    "label": int(lab),
                }
            )
        return pd.DataFrame(rows)
