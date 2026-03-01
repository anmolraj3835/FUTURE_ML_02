"""
preprocessing.py - Text Preprocessing Pipeline
Handles all NLP preprocessing: lowering, cleaning, stopword removal,
tokenization, and lemmatization using NLTK.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ─── Download required NLTK data (idempotent) ───────────────────────────────

_NLTK_DATA = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
for pkg in _NLTK_DATA:
    nltk.download(pkg, quiet=True)

# ─── Initialise reusable objects ─────────────────────────────────────────────

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


# ─── Individual preprocessing steps ─────────────────────────────────────────

def to_lowercase(text: str) -> str:
    """Step 1: Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Step 2: Remove punctuation and special characters."""
    # Keep only alphabetic chars, digits and spaces
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Step 3: Remove English stopwords from a list of tokens."""
    return [t for t in tokens if t not in _stop_words]


def tokenize(text: str) -> list[str]:
    """Step 4: Tokenize text into word tokens."""
    return word_tokenize(text)


def lemmatize(tokens: list[str]) -> list[str]:
    """Step 5: Lemmatize each token to its base form."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


# ─── Full pipeline ──────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Run the full preprocessing pipeline on a single text string.

    Pipeline:
        lowercase → remove punctuation → tokenize → remove stopwords → lemmatize
    Returns a single cleaned string (tokens joined by space).
    """
    text = to_lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_series(series):
    """Apply the preprocessing pipeline to a pandas Series."""
    return series.apply(preprocess)
