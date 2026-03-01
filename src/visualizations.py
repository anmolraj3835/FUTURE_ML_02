"""
visualizations.py - Charts and plots for model evaluation
Uses Matplotlib and Seaborn for publication-quality figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ─── Style defaults ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = ["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#3b82f6"]


def plot_category_distribution(df, col="category", title="Ticket Category Distribution"):
    """Bar chart of category frequencies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[col].value_counts()
    sns.barplot(x=counts.values, y=counts.index, palette=COLORS, ax=ax)
    ax.set_xlabel("Number of Tickets")
    ax.set_title(title, fontweight="bold")
    for i, v in enumerate(counts.values):
        ax.text(v + 5, i, str(v), va="center", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_priority_distribution(df, col="priority", title="Priority Distribution"):
    """Pie chart of priority levels."""
    fig, ax = plt.subplots(figsize=(6, 6))
    counts = df[col].value_counts()
    colors_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
    colors = [colors_map.get(p, "#94a3b8") for p in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=140, textprops={"fontweight": "bold"},
    )
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None, display_labels=None, title="Confusion Matrix"):
    """Heatmap confusion matrix. Use display_labels for tick labels when y values are encoded."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tick_labels = display_labels if display_labels is not None else (labels if labels is not None else "auto")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=tick_labels,
                yticklabels=tick_labels, ax=ax)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_df):
    """Grouped bar chart comparing model metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(comparison_df))
    width = 0.18

    for i, metric in enumerate(metrics):
        vals = comparison_df[metric].astype(float).values
        bars = ax.bar(x + i * width, vals, width, label=metric, color=COLORS[i])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df["Model"], fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def plot_text_length_distribution(df, text_col="text"):
    """Histogram of ticket text lengths."""
    fig, ax = plt.subplots(figsize=(8, 5))
    df["_len"] = df[text_col].str.split().str.len()
    sns.histplot(data=df, x="_len", hue="category", kde=True,
                 palette=COLORS, ax=ax, alpha=0.6)
    ax.set_xlabel("Number of Words")
    ax.set_title("Ticket Length Distribution by Category", fontweight="bold")
    df.drop(columns=["_len"], inplace=True, errors="ignore")
    plt.tight_layout()
    return fig
