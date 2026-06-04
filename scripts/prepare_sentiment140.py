#!/usr/bin/env python3
"""
Prepare a subset of Sentiment140 for training.

Download training.6M.csv from:
https://www.kaggle.com/datasets/kazanova/sentiment140
or the original Sentiment140 archive, then run:

  python scripts/prepare_sentiment140.py --input data/raw/training.6M.csv --samples 50000
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import normalize_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Sentiment140 for faster training")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/sentiment140_sample.csv")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(args.input, encoding="latin-1", header=None, names=names)
    df["sentiment"] = normalize_labels(df["target"])
    df = df[df["sentiment"].isin([0, 1])]

    sample = df.sample(n=min(args.samples, len(df)), random_state=args.seed)
    out = sample[["text", "sentiment"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
