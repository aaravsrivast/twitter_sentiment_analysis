#!/usr/bin/env python3
"""Download NLTK data (handles macOS SSL certificate issues)."""

import ssl

import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for package in (
    "stopwords",
    "wordnet",
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "omw-1.4",
):
    nltk.download(package)
    print(f"Downloaded: {package}")

print("NLTK setup complete.")
