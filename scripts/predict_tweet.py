#!/usr/bin/env python3
"""Predict sentiment for a single tweet or text file."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference.predict import SentimentPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict tweet sentiment")
    parser.add_argument("text", nargs="?", help="Tweet text to classify")
    parser.add_argument("--file", type=Path, help="File with one tweet per line")
    parser.add_argument("--model", default=None, help="Model name (default from config)")
    args = parser.parse_args()

    config = load_config()
    model_name = args.model or config["training"]["default_model"]
    predictor = SentimentPredictor(model_name=model_name)

    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        texts.extend(args.file.read_text(encoding="utf-8").strip().splitlines())

    if not texts:
        parser.error("Provide text argument or --file")

    for t in texts:
        result = predictor.predict_one(t)
        conf = result.get("confidence") or {}
        conf_str = ", ".join(f"{k}: {v:.2%}" for k, v in conf.items()) if conf else "N/A"
        print(f"\nTweet: {result['text']}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {conf_str}")


if __name__ == "__main__":
    main()
