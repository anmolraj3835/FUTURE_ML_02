"""
feature_extraction.py - TF-IDF and Bag-of-Words feature extraction
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def build_tfidf(train_texts, max_features: int = 5000):
    """
    Fit a TF-IDF vectorizer on training texts and return (vectorizer, matrix).
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(train_texts)
    return vectorizer, X


def build_bow(train_texts, max_features: int = 5000):
    """
    Fit a Bag-of-Words (Count) vectorizer on training texts and return (vectorizer, matrix).
    """
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(train_texts)
    return vectorizer, X
