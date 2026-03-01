"""
priority.py - Priority Assignment Logic
Implements both rule-based and model-based priority prediction.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ─── High-priority keywords ─────────────────────────────────────────────────

HIGH_KEYWORDS = [
    "urgent", "error", "failed", "not working", "immediately",
    "critical", "crashed", "broken", "lost", "hacked",
    "unauthorized", "locked out", "suspended", "fatal", "asap",
    "emergency",
]

LOW_INDICATORS = [
    "information", "pricing", "plan", "how do", "what is",
    "features", "available", "trial", "documentation", "tutorial",
    "demo", "feedback", "roadmap", "inquiry",
]


def rule_based_priority(text: str) -> str:
    """
    Assign priority using keyword-based rules.

    - High   → contains urgent / error / failure keywords
    - Low    → general inquiry language
    - Medium → everything else
    """
    lower = text.lower()
    for kw in HIGH_KEYWORDS:
        if kw in lower:
            return "High"
    for kw in LOW_INDICATORS:
        if kw in lower:
            return "Low"
    return "Medium"


def train_priority_model(X_train, y_train):
    """Train a Logistic Regression model for priority prediction."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_priority_model(model, X_test, y_test) -> dict:
    """Evaluate the priority model and return metrics."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "y_pred": y_pred,
    }
