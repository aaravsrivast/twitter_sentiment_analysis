"""NLP preprocessing for Twitter text."""

import re
import ssl
from typing import Iterable

import nltk
import pandas as pd

CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
}

# Fallback when NLTK corpora are unavailable (e.g. SSL errors on download)
FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "as", "by", "with", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "our", "their", "what", "which", "who",
    "whom", "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "don", "now", "rt", "via",
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

_nltk_ready = False
_use_nltk_tokenizer = False
_use_lemmatizer = False


def ensure_nltk_data() -> None:
    global _nltk_ready, _use_nltk_tokenizer, _use_lemmatizer
    if _nltk_ready:
        return

    resources = [
        "stopwords",
        "wordnet",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "omw-1.4",
    ]
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    for name in resources:
        try:
            nltk.download(name, quiet=True)
        except Exception:
            pass

    try:
        from nltk.corpus import stopwords as nltk_stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        nltk_stopwords.words("english")
        word_tokenize("test sentence")
        WordNetLemmatizer().lemmatize("running")
        _use_nltk_tokenizer = True
        _use_lemmatizer = True
    except Exception:
        _use_nltk_tokenizer = False
        _use_lemmatizer = False

    _nltk_ready = True


def _tokenize(text: str) -> list[str]:
    if _use_nltk_tokenizer:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    return WHITESPACE_PATTERN.split(text.strip())


def _lemmatize_tokens(tokens: list[str]) -> list[str]:
    if _use_lemmatizer:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def _get_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return FALLBACK_STOPWORDS


class TextCleaner:
    """Clean and normalize tweet text for ML feature extraction."""

    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        expand_contractions: bool = True,
        use_lemmatization: bool = True,
    ) -> None:
        ensure_nltk_data()
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.expand_contractions = expand_contractions
        self.use_lemmatization = use_lemmatization and _use_lemmatizer
        self._stopwords = _get_stopwords()

    def _expand_contractions(self, text: str) -> str:
        lowered = text.lower()
        for short, full in CONTRACTIONS.items():
            lowered = lowered.replace(short, full)
        return lowered

    def clean(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.lower()
        if self.expand_contractions:
            text = self._expand_contractions(text)
        if self.remove_urls:
            text = URL_PATTERN.sub(" ", text)
        if self.remove_mentions:
            text = MENTION_PATTERN.sub(" ", text)
        if self.remove_hashtags:
            text = HASHTAG_PATTERN.sub(" ", text)
        else:
            text = HASHTAG_PATTERN.sub(r"\1", text)

        text = NON_ALPHA_PATTERN.sub(" ", text)
        tokens = _tokenize(text)
        tokens = [t for t in tokens if t not in self._stopwords and len(t) > 1]

        if self.use_lemmatization:
            tokens = _lemmatize_tokens(tokens)

        return " ".join(tokens)

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).map(self.clean)

    def transform_iterable(self, texts: Iterable[str]) -> list[str]:
        return [self.clean(t) for t in texts]
