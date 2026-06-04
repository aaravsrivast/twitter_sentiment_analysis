#!/usr/bin/env python3
"""Fetch recent tweets and run sentiment trend analysis."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.trends import SentimentTrendAnalyzer
from src.config import get_path, load_config
from src.twitter.client import TwitterClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch tweets and analyze sentiment")
    parser.add_argument("query", help='Search query, e.g. "#machinelearning lang:en"')
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = load_config()
    outputs_dir = get_path("outputs_dir", config)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    client = TwitterClient()
    tweets = client.search_recent(args.query, max_results=args.max_results)
    if tweets.empty:
        print("No tweets found for query.")
        return

    from src.inference.predict import SentimentPredictor

    model_name = args.model or config["training"]["default_model"]
    analyzer = SentimentTrendAnalyzer(SentimentPredictor(model_name=model_name))
    analyzed = analyzer.analyze_dataframe(tweets)

    run_dir = outputs_dir / "latest_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    analyzed.to_csv(run_dir / "analyzed_tweets.csv", index=False)

    summary = analyzer.summary_report(analyzed)
    analyzer.plot_distribution(analyzed, run_dir / "sentiment_distribution.png")
    analyzer.plot_timeline(analyzed, run_dir / "sentiment_timeline.png")
    for sentiment in ("positive", "negative"):
        subset = analyzed[analyzed["sentiment"] == sentiment]
        if len(subset) >= 5:
            analyzer.plot_wordcloud(analyzed, sentiment, run_dir / f"wordcloud_{sentiment}.png")

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to {run_dir}/")


if __name__ == "__main__":
    main()
