#!/usr/bin/env python3
"""Stream live tweets and print real-time sentiment (requires filtered stream access)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference.predict import SentimentPredictor
from src.twitter.client import TwitterClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream tweets with live sentiment")
    parser.add_argument(
        "--rule",
        action="append",
        required=True,
        help='Stream rule (repeatable), e.g. "#AI lang:en"',
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = load_config()
    model_name = args.model or config["training"]["default_model"]
    predictor = SentimentPredictor(model_name=model_name)

    def on_tweet(tweet: dict) -> None:
        result = predictor.predict_one(tweet["text"])
        print(f"[{result['sentiment'].upper()}] {tweet['text'][:120]}")

    client = TwitterClient()
    print("Starting filtered stream (Ctrl+C to stop)...")
    client.stream_filter(args.rule, on_tweet)


if __name__ == "__main__":
    main()
