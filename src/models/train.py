"""Train and evaluate sentiment classifiers."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import load_config
from src.data.loader import load_dataset
from src.features.vectorizer import fit_vectorizer, save_vectorizer
from src.preprocessing.text_cleaner import TextCleaner

MODEL_REGISTRY: dict[str, Any] = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "multinomial_nb": MultinomialNB(),
    "linear_svc": LinearSVC(class_weight="balanced", random_state=42),
}


def _build_cleaner(config: dict[str, Any]) -> TextCleaner:
    p = config.get("preprocessing", {})
    return TextCleaner(
        remove_urls=p.get("remove_urls", True),
        remove_mentions=p.get("remove_mentions", True),
        remove_hashtags=p.get("remove_hashtags", False),
        expand_contractions=p.get("expand_contractions", True),
        use_lemmatization=p.get("use_lemmatization", True),
    )


def prepare_features(
    df: pd.DataFrame,
    config: dict[str, Any],
    cleaner: TextCleaner | None = None,
) -> tuple[list[str], np.ndarray]:
    cleaner = cleaner or _build_cleaner(config)
    cleaned = cleaner.transform_series(df["text"])
    return cleaned.tolist(), df["sentiment"].values


def train_and_evaluate(
    data_path: Path,
    config: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    seed = config.get("project", {}).get("random_seed", 42)
    train_cfg = config.get("training", {})
    model_name = model_name or train_cfg.get("default_model", "logistic_regression")

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY)}")

    df = load_dataset(data_path)
    texts, y = prepare_features(df, config)
    vectorizer = fit_vectorizer(texts, config)
    X = vectorizer.transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_cfg.get("test_size", 0.2),
        random_state=seed,
        stratify=y,
    )

    clf = MODEL_REGISTRY[model_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cv_folds = train_cfg.get("cv_folds", 5)
    cv_scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="f1")

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["negative", "positive"]
        ),
        "n_samples": len(df),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
    }
    return {
        "classifier": clf,
        "vectorizer": vectorizer,
        "cleaner_config": config.get("preprocessing", {}),
        "metrics": metrics,
    }


def save_artifacts(
    result: dict[str, Any],
    models_dir: Path,
    model_name: str,
) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    prefix = models_dir / model_name

    joblib.dump(result["classifier"], f"{prefix}_classifier.joblib")
    save_vectorizer(result["vectorizer"], Path(f"{prefix}_vectorizer.joblib"))
    joblib.dump(result["metrics"], f"{prefix}_metrics.joblib")
    return prefix


def load_classifier(models_dir: Path, model_name: str):
    models_dir = Path(models_dir)
    prefix = models_dir / model_name
    return joblib.load(f"{prefix}_classifier.joblib")
