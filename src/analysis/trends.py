"""Sentiment trend analysis and visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.inference.predict import SentimentPredictor


class SentimentTrendAnalyzer:
    def __init__(self, predictor: SentimentPredictor | None = None) -> None:
        self.predictor = predictor or SentimentPredictor()

    def analyze_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        preds = self.predictor.predict_batch(df[text_col].tolist())
        out = df.copy().reset_index(drop=True)
        out["sentiment"] = preds["sentiment"].values
        out["label"] = preds["label"].values
        out["cleaned_text"] = preds["cleaned_text"].values
        if "created_at" in out.columns:
            out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
        return out

    def sentiment_distribution(self, df: pd.DataFrame) -> pd.Series:
        return df["sentiment"].value_counts()

    def plot_distribution(self, df: pd.DataFrame, output_path: Path) -> Path:
        counts = self.sentiment_distribution(df)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"positive": "#2ecc71", "negative": "#e74c3c"}
        sns.barplot(
            x=counts.index,
            y=counts.values,
            hue=counts.index,
            palette=[colors.get(i, "#3498db") for i in counts.index],
            legend=False,
            ax=ax,
        )
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def plot_timeline(self, df: pd.DataFrame, output_path: Path) -> Path | None:
        if "created_at" not in df.columns or df["created_at"].isna().all():
            return None

        timeline = (
            df.dropna(subset=["created_at"])
            .set_index("created_at")
            .resample("1h")["label"]
            .mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(timeline["created_at"], timeline["label"], marker="o", linewidth=2)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Neutral")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Sentiment Trend Over Time (1 = positive)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean sentiment score")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def plot_wordcloud(
        self,
        df: pd.DataFrame,
        sentiment: str,
        output_path: Path,
    ) -> Path:
        subset = df[df["sentiment"] == sentiment]
        text = " ".join(subset["cleaned_text"].dropna().astype(str))
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {sentiment.title()} Tweets")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def summary_report(self, df: pd.DataFrame) -> dict:
        total = len(df)
        dist = self.sentiment_distribution(df)
        positive_pct = 100 * dist.get("positive", 0) / total if total else 0
        negative_pct = 100 * dist.get("negative", 0) / total if total else 0
        return {
            "total_tweets": total,
            "positive_count": int(dist.get("positive", 0)),
            "negative_count": int(dist.get("negative", 0)),
            "positive_percent": round(positive_pct, 2),
            "negative_percent": round(negative_pct, 2),
        }
