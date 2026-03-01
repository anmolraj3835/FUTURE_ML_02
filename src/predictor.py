"""
predictor.py - Final Prediction Interface
Provides the predict_ticket() function and model persistence utilities.
"""

import os
import joblib
import pandas as pd
from src.preprocessing import preprocess
from src.priority import rule_based_priority


# ─── Model persistence ──────────────────────────────────────────────────────

MODELS_DIR = "saved_models"


def save_artifacts(model, vectorizer, label_encoder=None, directory: str = MODELS_DIR):
    """Save trained model, vectorizer, and optional label encoder to disk."""
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model, os.path.join(directory, "best_model.pkl"))
    joblib.dump(vectorizer, os.path.join(directory, "tfidf_vectorizer.pkl"))
    if label_encoder is not None:
        joblib.dump(label_encoder, os.path.join(directory, "label_encoder.pkl"))
    print(f"✅ Artifacts saved to '{directory}/'")


def load_artifacts(directory: str = MODELS_DIR):
    """Load model and vectorizer from disk."""
    model = joblib.load(os.path.join(directory, "best_model.pkl"))
    vectorizer = joblib.load(os.path.join(directory, "tfidf_vectorizer.pkl"))
    le_path = os.path.join(directory, "label_encoder.pkl")
    label_encoder = joblib.load(le_path) if os.path.exists(le_path) else None
    return model, vectorizer, label_encoder


# ─── Prediction function ────────────────────────────────────────────────────

# Module-level cache so we only load once
_cached = {"model": None, "vectorizer": None, "le": None}


def predict_ticket(text: str, model=None, vectorizer=None, label_encoder=None) -> tuple:
    """
    Predict the category and priority for a customer support ticket.

    Parameters
    ----------
    text : str
        Raw ticket text.
    model : sklearn estimator, optional
        If None, loads the saved model from disk.
    vectorizer : TfidfVectorizer, optional
        If None, loads from disk.
    label_encoder : LabelEncoder, optional
        If None, loads from disk (if available).

    Returns
    -------
    (category: str, priority: str)
    """
    # Load from cache / disk if not provided
    if model is None or vectorizer is None:
        if _cached["model"] is None:
            _cached["model"], _cached["vectorizer"], _cached["le"] = load_artifacts()
        model = model or _cached["model"]
        vectorizer = vectorizer or _cached["vectorizer"]
        label_encoder = label_encoder or _cached["le"]

    # Preprocess → vectorize → predict category
    cleaned = preprocess(text)
    X = vectorizer.transform([cleaned])
    category = model.predict(X)[0]

    # Decode label if encoder was used
    if label_encoder is not None:
        category = label_encoder.inverse_transform([category])[0]

    # Assign priority (rule-based on raw text)
    priority = rule_based_priority(text)

    return category, priority


def predict_batch(texts: list[str], **kwargs) -> pd.DataFrame:
    """Run predict_ticket on a list of texts and return a DataFrame."""
    results = [predict_ticket(t, **kwargs) for t in texts]
    return pd.DataFrame(results, columns=["category", "priority"])


def export_predictions(df: pd.DataFrame, path: str = "data/predictions.csv"):
    """Export a predictions DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Predictions exported to '{path}'")
    return path
