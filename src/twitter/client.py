"""Fetch live tweets via Tweepy (Twitter API v2)."""

import os
from datetime import datetime
from typing import Any, Callable

import pandas as pd
import tweepy
from dotenv import load_dotenv

load_dotenv()

SENTIMENT_LABELS = {0: "negative", 1: "positive"}


class TwitterClient:
    """Search and stream tweets for real-time sentiment analysis."""

    def __init__(self, bearer_token: str | None = None) -> None:
        token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        if not token:
            raise ValueError(
                "TWITTER_BEARER_TOKEN is required. Copy .env.example to .env and add credentials."
            )
        self.client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)

    def search_recent(
        self,
        query: str,
        max_results: int = 100,
        tweet_fields: list[str] | None = None,
    ) -> pd.DataFrame:
        tweet_fields = tweet_fields or ["created_at", "lang", "public_metrics"]
        max_results = min(max(max_results, 10), 100)

        response = self.client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=tweet_fields,
        )
        if not response.data:
            return pd.DataFrame(columns=["id", "text", "created_at", "author_id"])

        rows = []
        for tweet in response.data:
            rows.append(
                {
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "author_id": getattr(tweet, "author_id", None),
                    "lang": getattr(tweet, "lang", None),
                    "retweet_count": getattr(tweet, "public_metrics", {}).get("retweet_count")
                    if hasattr(tweet, "public_metrics") and tweet.public_metrics
                    else None,
                    "like_count": getattr(tweet, "public_metrics", {}).get("like_count")
                    if hasattr(tweet, "public_metrics") and tweet.public_metrics
                    else None,
                }
            )
        return pd.DataFrame(rows)

    def stream_filter(
        self,
        rules: list[str],
        on_tweet: Callable[[dict[str, Any]], None],
        bearer_token: str | None = None,
    ) -> None:
        """
        Stream filtered tweets (requires Elevated access + filtered stream).

        rules: list of stream rule strings, e.g. ["#AI lang:en"]
        on_tweet: callback receiving {id, text, created_at}
        """
        token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        if not token:
            raise ValueError("TWITTER_BEARER_TOKEN is required for streaming.")

        class StreamHandler(tweepy.StreamingClient):
            def on_tweet(self, tweet):
                on_tweet(
                    {
                        "id": tweet.id,
                        "text": tweet.text,
                        "created_at": datetime.utcnow().isoformat(),
                    }
                )

        stream = StreamHandler(bearer_token=token)
        existing = stream.get_rules()
        if existing.data:
            stream.delete_rules([r.id for r in existing.data])

        stream.add_rules([tweepy.StreamRule(rule) for rule in rules])
        stream.filter(tweet_fields=["created_at", "lang"])
