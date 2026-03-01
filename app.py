"""
app.py - Flask Web Application
Customer Support Ticket Classification & Priority Assignment
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify
from src.predictor import predict_ticket, load_artifacts
from src.priority import rule_based_priority
from src.preprocessing import preprocess

app = Flask(__name__, template_folder="templates", static_folder="static")

# Pre-load model at startup
print("Loading model artifacts...")
model, vectorizer, label_encoder = load_artifacts()
print("Model loaded successfully!")


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for ticket prediction."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter a ticket description."}), 400

    try:
        category, priority = predict_ticket(
            text, model=model, vectorizer=vectorizer, label_encoder=label_encoder
        )

        # Get cleaned text for display
        cleaned = preprocess(text)

        # Get confidence scores
        cleaned_vec = vectorizer.transform([cleaned])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(cleaned_vec)[0]
            classes = label_encoder.inverse_transform(range(len(proba)))
            confidence = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}
        else:
            confidence = {category: 100.0}

        return jsonify({
            "category": category,
            "priority": priority,
            "confidence": confidence,
            "cleaned_text": cleaned,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch", methods=["POST"])
def batch_predict():
    """API endpoint for batch ticket prediction."""
    data = request.get_json()
    tickets = data.get("tickets", [])

    if not tickets:
        return jsonify({"error": "No tickets provided."}), 400

    results = []
    for text in tickets:
        text = text.strip()
        if text:
            category, priority = predict_ticket(
                text, model=model, vectorizer=vectorizer, label_encoder=label_encoder
            )
            results.append({
                "text": text,
                "category": category,
                "priority": priority,
            })

    return jsonify({"results": results})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Ticket Classification System - Live Server")
    print("=" * 60)
    print("  Open in browser: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
