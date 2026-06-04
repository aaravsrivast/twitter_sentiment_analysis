#!/usr/bin/env python3
"""Train sentiment classifiers and save artifacts."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_path, load_config
from src.models.train import MODEL_REGISTRY, save_artifacts, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Twitter sentiment model")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to CSV dataset (default: sample data)",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help="Classifier to train",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train all models and print comparison",
    )
    args = parser.parse_args()

    config = load_config()
    data_path = args.data or get_path("sample_data", config)
    models_dir = get_path("models_dir", config)
    outputs_dir = get_path("outputs_dir", config)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        results = []
        for name in MODEL_REGISTRY:
            result = train_and_evaluate(data_path, config, model_name=name)
            save_artifacts(result, models_dir, name)
            m = result["metrics"]
            results.append(
                {
                    "model": name,
                    "accuracy": m["accuracy"],
                    "f1": m["f1"],
                    "cv_f1_mean": m["cv_f1_mean"],
                }
            )
            print(f"\n=== {name} ===")
            print(m["classification_report"])

        best = max(results, key=lambda x: x["f1"])
        print("\n--- Model comparison ---")
        for r in results:
            marker = " <-- best" if r["model"] == best["model"] else ""
            print(f"{r['model']}: acc={r['accuracy']:.4f} f1={r['f1']:.4f} cv_f1={r['cv_f1_mean']:.4f}{marker}")

        with open(outputs_dir / "model_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved comparison to {outputs_dir / 'model_comparison.json'}")
        return

    model_name = args.model or config["training"]["default_model"]
    result = train_and_evaluate(data_path, config, model_name=model_name)
    save_artifacts(result, models_dir, model_name)

    metrics = result["metrics"]
    print(f"Trained: {model_name}")
    print(f"Samples: {metrics['n_samples']} (train={metrics['n_train']}, test={metrics['n_test']})")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"CV F1: {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})")
    print(metrics["classification_report"])
    print(f"\nArtifacts saved to {models_dir}/")


if __name__ == "__main__":
    main()
