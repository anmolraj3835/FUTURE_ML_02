"""
=======================================================================
  Customer Support Ticket Classification & Priority Assignment System
=======================================================================
  End-to-end NLP pipeline that classifies support tickets into
  categories and assigns priority levels automatically.

  Run:  python main.py
=======================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Fix Windows terminal encoding
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for terminal execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Project modules
from src.dataset import generate_dataset, save_dataset
from src.preprocessing import preprocess, preprocess_series
from src.preprocessing import to_lowercase, remove_punctuation, tokenize, remove_stopwords, lemmatize
from src.feature_extraction import build_tfidf, build_bow
from src.models import get_models, train_model, evaluate_model, compare_models, select_best_model
from src.priority import rule_based_priority, train_priority_model, evaluate_priority_model
from src.visualizations import (
    plot_category_distribution, plot_priority_distribution,
    plot_confusion_matrix, plot_model_comparison, plot_text_length_distribution,
)
from src.predictor import predict_ticket, save_artifacts, export_predictions, predict_batch


# ─── Helper: section headers ────────────────────────────────────────────────

def header(title, char="=", width=70):
    """Print a styled section header."""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def sub_header(title, char="-", width=70):
    """Print a styled sub-section header."""
    print()
    print(f"  {title}")
    print(char * width)


# ─── Ensure output directories exist ────────────────────────────────────────

os.makedirs("data", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
os.makedirs("output_charts", exist_ok=True)


# =========================================================================
#  STEP 1 : Dataset Generation & EDA
# =========================================================================

header("STEP 1 : Dataset Generation & Exploratory Data Analysis")

df = generate_dataset(n_samples=1000, seed=42)
save_dataset(df, "data/tickets.csv")

print(f"\n  Dataset shape : {df.shape}")
print(f"  Columns       : {list(df.columns)}")

sub_header("First 10 Tickets")
print(df.head(10).to_string(index=False))

sub_header("Category Distribution")
cat_counts = df["category"].value_counts()
for cat, count in cat_counts.items():
    bar = "█" * (count // 10)
    print(f"  {cat:<20} {count:>4}  {bar}")

sub_header("Priority Distribution")
pri_counts = df["priority"].value_counts()
for pri, count in pri_counts.items():
    bar = "█" * (count // 10)
    print(f"  {pri:<10} {count:>4}  {bar}")

# Save charts
fig = plot_category_distribution(df)
fig.savefig("output_charts/category_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig = plot_priority_distribution(df)
fig.savefig("output_charts/priority_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig = plot_text_length_distribution(df)
fig.savefig("output_charts/text_length_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n  [Charts saved to output_charts/]")


# =========================================================================
#  STEP 2 : Text Preprocessing
# =========================================================================

header("STEP 2 : Text Preprocessing")

print("""
  Pipeline:
    1. Convert to lowercase
    2. Remove punctuation & special characters
    3. Tokenization  (NLTK word_tokenize)
    4. Stopword removal
    5. Lemmatization  (WordNetLemmatizer)
""")

# Demonstrate on a single sample
sample = df["text"].iloc[0]
print(f"  Original      : {sample}")
s1 = to_lowercase(sample)
print(f"  Lowercase     : {s1}")
s2 = remove_punctuation(s1)
print(f"  No punctuation: {s2}")
s3 = tokenize(s2)
print(f"  Tokens        : {s3}")
s4 = remove_stopwords(s3)
print(f"  No stopwords  : {s4}")
s5 = lemmatize(s4)
print(f"  Lemmatized    : {s5}")
print(f"  Final string  : {' '.join(s5)}")

# Apply to full dataset
df["cleaned_text"] = preprocess_series(df["text"])
print("\n  Preprocessing applied to all 1,000 tickets.")

sub_header("Before vs After")
for i in range(5):
    print(f"  [{i}] BEFORE : {df['text'].iloc[i][:80]}")
    print(f"      AFTER  : {df['cleaned_text'].iloc[i][:80]}")
    print()


# =========================================================================
#  STEP 3 : Feature Extraction
# =========================================================================

header("STEP 3 : Feature Extraction (TF-IDF & Bag of Words)")

# Encode labels
le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df["category"])

print("\n  Label Mapping:")
for cls, idx in zip(le.classes_, le.transform(le.classes_)):
    print(f"    {idx} -> {cls}")

# Train / Test split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["cleaned_text"], df["category_encoded"],
    test_size=0.2, random_state=42, stratify=df["category_encoded"],
)
print(f"\n  Train size : {len(X_train_text)}")
print(f"  Test size  : {len(X_test_text)}")

# TF-IDF
tfidf_vec, X_train_tfidf = build_tfidf(X_train_text)
X_test_tfidf = tfidf_vec.transform(X_test_text)
print(f"\n  TF-IDF matrix : {X_train_tfidf.shape}")

# Bag of Words
bow_vec, X_train_bow = build_bow(X_train_text)
X_test_bow = bow_vec.transform(X_test_text)
print(f"  BoW matrix    : {X_train_bow.shape}")


# =========================================================================
#  STEP 4 : Model Training & Comparison
# =========================================================================

header("STEP 4 : Model Training & Comparison")

# ── Train on TF-IDF ──
sub_header("Training on TF-IDF features")
models = get_models()
results = {}

for name, model in models.items():
    print(f"\n  Training {name} ...")
    trained = train_model(model, X_train_tfidf, y_train)
    res = evaluate_model(trained, X_test_tfidf, y_test)
    results[name] = res
    results[name]["trained_model"] = trained
    print(f"    Accuracy : {res['accuracy']:.4f}")
    print(f"    F1-Score : {res['f1']:.4f}")

sub_header("TF-IDF Model Comparison")
comparison_df = compare_models(results)
print(comparison_df.to_string(index=False))

# ── Train on BoW ──
sub_header("Training on Bag-of-Words features")
bow_results = {}
for name, model_class in get_models().items():
    trained = train_model(model_class, X_train_bow, y_train)
    bow_results[name] = evaluate_model(trained, X_test_bow, y_test)

bow_comparison = compare_models(bow_results)

sub_header("Full Comparison : TF-IDF vs Bag of Words")
tfidf_comp = compare_models(results).copy()
tfidf_comp["Model"] = tfidf_comp["Model"] + " (TF-IDF)"
bow_comp = bow_comparison.copy()
bow_comp["Model"] = bow_comp["Model"] + " (BoW)"
full_comparison = pd.concat([tfidf_comp, bow_comp], ignore_index=True)
print(full_comparison.to_string(index=False))


# =========================================================================
#  STEP 5 : Model Evaluation & Visualizations
# =========================================================================

header("STEP 5 : Model Evaluation")

best_name = select_best_model(results)
best_result = results[best_name]
best_model = best_result["trained_model"]

print(f"\n  Best Model : {best_name}")
print(f"  F1-Score   : {best_result['f1']:.4f}")
print(f"  Accuracy   : {best_result['accuracy']:.4f}")

sub_header(f"Classification Report - {best_name}")
print(best_result["report"])

sub_header("Confusion Matrix (text)")
cm = best_result["confusion_matrix"]
labels = le.classes_
# Print header row
print(f"  {'':>20}", end="")
for lbl in labels:
    print(f"  {lbl:>16}", end="")
print()
for i, row_label in enumerate(labels):
    print(f"  {row_label:>20}", end="")
    for j in range(len(labels)):
        print(f"  {cm[i][j]:>16}", end="")
    print()

# Save confusion matrix chart
fig = plot_confusion_matrix(
    y_test, best_result["y_pred"], display_labels=le.classes_,
    title=f"Confusion Matrix - {best_name}",
)
fig.savefig("output_charts/confusion_matrix_best.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Save all-models confusion matrix chart
from sklearn.metrics import confusion_matrix as cm_func
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm_vals = cm_func(y_test, res["y_pred"])
    sns.heatmap(cm_vals, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices - All Models", fontweight="bold", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig("output_charts/confusion_matrices_all.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Save model comparison chart
chart_df = compare_models(results)
fig = plot_model_comparison(chart_df)
fig.savefig("output_charts/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n  [All charts saved to output_charts/]")


# =========================================================================
#  STEP 6 : Priority Assignment
# =========================================================================

header("STEP 6 : Priority Assignment")

# ── 6a. Rule-based ──
sub_header("6a. Rule-Based Priority")
print("""
  HIGH   -> Keywords: urgent, error, failed, not working, immediately,
            critical, crashed, broken, lost, hacked, unauthorized, ...
  LOW    -> Keywords: information, pricing, plan, how do, features,
            trial, documentation, tutorial, demo, ...
  MEDIUM -> Everything else
""")

df["predicted_priority_rule"] = df["text"].apply(rule_based_priority)
rule_acc = accuracy_score(df["priority"], df["predicted_priority_rule"])
print(f"  Rule-based accuracy : {rule_acc:.4f}")
print()
print(pd.crosstab(df["priority"], df["predicted_priority_rule"], margins=True).to_string())

# ── 6b. Model-based ──
sub_header("6b. Model-Based Priority (Logistic Regression)")
le_priority = LabelEncoder()
df["priority_encoded"] = le_priority.fit_transform(df["priority"])

_, _, y_train_pri, y_test_pri = train_test_split(
    df["cleaned_text"], df["priority_encoded"],
    test_size=0.2, random_state=42, stratify=df["priority_encoded"],
)

pri_model = train_priority_model(X_train_tfidf, y_train_pri)
pri_eval = evaluate_priority_model(pri_model, X_test_tfidf, y_test_pri)

print(f"\n  Model-based accuracy : {pri_eval['accuracy']:.4f}")
print()
print(pri_eval["report"])


# =========================================================================
#  STEP 7 : Save Model Artifacts
# =========================================================================

header("STEP 7 : Save Model Artifacts")

save_artifacts(best_model, tfidf_vec, le)

print("  Saved files:")
for f in os.listdir("saved_models"):
    size = os.path.getsize(os.path.join("saved_models", f))
    print(f"    saved_models/{f:<25} ({size:,} bytes)")


# =========================================================================
#  STEP 8 : Prediction Function & Test Cases
# =========================================================================

header("STEP 8 : Prediction Function & Test Cases")

print("""
  Function signature:
    def predict_ticket(text: str) -> tuple[str, str]:
        return (category, priority)
""")

test_tickets = [
    # Technical Issues
    "URGENT: My app crashed and I lost all my data!",
    "The software keeps freezing when I try to save files.",
    # Billing Issues
    "I was charged twice on my credit card this month.",
    "I need a refund immediately, the payment was unauthorized!",
    # Account Access
    "I can't log into my account, the password reset is broken.",
    "My account was locked and I need help getting back in.",
    # General Inquiry
    "What subscription plans do you offer?",
    "Do you have documentation available for the API?",
    # Edge cases
    "This is not working and I need it fixed immediately!",
    "Just wondering about your pricing.",
]

print(f"  {'#':<4} {'TICKET TEXT':<55} {'CATEGORY':<20} {'PRIORITY':<10}")
print("  " + "-" * 89)

for i, ticket in enumerate(test_tickets, 1):
    category, priority = predict_ticket(ticket)
    short = ticket[:52] + "..." if len(ticket) > 55 else ticket
    print(f"  {i:<4} {short:<55} {category:<20} {priority:<10}")

print("  " + "-" * 89)


# =========================================================================
#  STEP 9 : Export Predictions to CSV
# =========================================================================

header("STEP 9 : Export Predictions to CSV")

export_df = pd.DataFrame({
    "ticket_id": df.loc[X_test_text.index, "ticket_id"].values,
    "original_text": df.loc[X_test_text.index, "text"].values,
    "actual_category": le.inverse_transform(y_test),
    "predicted_category": le.inverse_transform(best_result["y_pred"]),
    "actual_priority": df.loc[X_test_text.index, "priority"].values,
    "predicted_priority": df.loc[X_test_text.index, "text"].apply(rule_based_priority).values,
})

export_predictions(export_df, "data/predictions.csv")

print(f"\n  Exported {len(export_df)} predictions to data/predictions.csv")
print()
print("  Preview (first 10 rows):")
print(export_df.head(10).to_string(index=False))


# =========================================================================
#  FINAL SUMMARY
# =========================================================================

header("PROJECT SUMMARY", char="*")

print(f"""
  Dataset size         : {len(df)} tickets
  Categories           : {list(le.classes_)}
  Priority levels      : High, Medium, Low

  Best model           : {best_name}
  Best Accuracy        : {best_result['accuracy']:.4f}
  Best F1-Score        : {best_result['f1']:.4f}
  Rule-based priority  : {rule_acc:.4f} accuracy

  Saved artifacts:
    saved_models/best_model.pkl
    saved_models/tfidf_vectorizer.pkl
    saved_models/label_encoder.pkl

  Exported files:
    data/tickets.csv           (synthetic dataset)
    data/predictions.csv       (test predictions)

  Charts:
    output_charts/category_distribution.png
    output_charts/priority_distribution.png
    output_charts/text_length_distribution.png
    output_charts/confusion_matrix_best.png
    output_charts/confusion_matrices_all.png
    output_charts/model_comparison.png
""")

print("=" * 70)
print("  Pipeline complete!")
print("=" * 70)
