"""
models.py - Model Training, Evaluation, and Comparison
Trains Logistic Regression, Naive Bayes, and Random Forest classifiers.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


# ─── Model definitions ──────────────────────────────────────────────────────

def get_models() -> dict:
    """Return a dict of model_name → untrained estimator."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }


# ─── Training ───────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train):
    """Fit a model and return it."""
    model.fit(X_train, y_train)
    return model


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test) -> dict:
    """Return a dict with accuracy, precision, recall, f1, and predictions."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "y_pred": y_pred,
        "report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def compare_models(results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame from results dict {name: eval_dict}."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{res['accuracy']:.4f}",
            "Precision": f"{res['precision']:.4f}",
            "Recall": f"{res['recall']:.4f}",
            "F1-Score": f"{res['f1']:.4f}",
        })
    return pd.DataFrame(rows)


def select_best_model(results: dict) -> str:
    """Return the name of the model with the highest F1 score."""
    return max(results, key=lambda k: results[k]["f1"])
