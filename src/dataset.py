"""
dataset.py - Synthetic Dataset Generator
Generates a realistic customer support ticket dataset with categories and priorities.
"""

import pandas as pd
import numpy as np
import random
import os

# ─── Ticket templates per category ───────────────────────────────────────────

TEMPLATES = {
    "Technical Issue": [
        "My application keeps crashing every time I try to open it.",
        "The software is not working after the latest update.",
        "I'm getting an error message when I try to log in.",
        "The system is extremely slow and keeps freezing.",
        "I cannot install the latest version on my computer.",
        "Error 503 keeps appearing when accessing the dashboard.",
        "The app crashes immediately on startup with a fatal error.",
        "Screen goes blank when I try to upload a file.",
        "The integration with third-party tools has stopped working.",
        "Push notifications are not being delivered to my device.",
        "My data synchronization failed and I lost recent changes.",
        "The search feature returns no results even for valid queries.",
        "Video calls keep dropping after a few minutes.",
        "The export to PDF function is broken and produces empty files.",
        "I'm experiencing frequent timeout errors on the server.",
        "The mobile app is not compatible with my operating system.",
        "Auto-save feature is not working and I lost my document.",
        "Cannot connect to the database, getting connection refused error.",
        "The website shows a 404 error on multiple pages.",
        "Performance has degraded significantly since the last patch.",
        "The API endpoint returns a 500 internal server error.",
        "Plugin compatibility issue after upgrading the platform.",
        "Two-factor authentication codes are not being received.",
        "File upload fails with a network error message.",
        "The calendar sync is broken between devices.",
    ],
    "Billing Issue": [
        "I was charged twice for my subscription this month.",
        "My payment failed but the amount was deducted from my account.",
        "I want a refund for the service I did not use.",
        "The invoice amount does not match my subscription plan.",
        "I need to update my payment method on file.",
        "Why was I charged after I cancelled my subscription?",
        "I see an unauthorized charge on my credit card statement.",
        "The promo code is not applying the discount at checkout.",
        "My free trial ended early and I was charged immediately.",
        "I need a detailed breakdown of my latest invoice.",
        "The pricing on the website differs from what I was charged.",
        "Can I get a receipt for my recent payment?",
        "I was charged for a premium feature I never activated.",
        "My automatic payment did not go through this cycle.",
        "I want to downgrade my plan but still being charged full price.",
        "Currency conversion fee was not mentioned during purchase.",
        "I need to dispute a charge that appears on my statement.",
        "The annual billing amount is incorrect.",
        "Refund has not been processed after 14 business days.",
        "Tax amount on invoice seems incorrect for my region.",
        "I cannot find the cancel subscription option anywhere.",
        "Charged for additional users that were never added.",
        "Payment gateway keeps rejecting my valid credit card.",
        "I need to split my invoice for different departments.",
        "Subscription renewed without sending any reminder email.",
    ],
    "Account Access": [
        "I forgot my password and the reset link is not working.",
        "My account has been locked after multiple failed attempts.",
        "I cannot access my account even with correct credentials.",
        "Someone may have hacked into my account without permission.",
        "I need to change the email address associated with my account.",
        "The verification email was never delivered to my inbox.",
        "My account was suspended without any prior notice.",
        "I want to enable two-factor authentication on my account.",
        "I cannot log in from my new device.",
        "My session keeps expiring every few minutes.",
        "I need to recover my account after losing my phone.",
        "The single sign-on integration is not working for my team.",
        "I'm locked out of my admin panel completely.",
        "Password reset token has expired before I could use it.",
        "My account permissions were changed without my knowledge.",
        "I cannot access shared resources that were assigned to me.",
        "Login page shows a blank screen on my browser.",
        "I need to merge two duplicate accounts into one.",
        "My account profile information cannot be updated.",
        "Guest access link is not working for external users.",
        "OAuth login with Google keeps failing with an error.",
        "I need to transfer account ownership to another person.",
        "Multi-device login limit is preventing me from working.",
        "Account deactivated even though I am an active subscriber.",
        "Cannot add team members to my organization account.",
    ],
    "General Inquiry": [
        "What are the available subscription plans and pricing?",
        "How do I get started with the platform?",
        "Can you provide more information about your enterprise plan?",
        "What features are included in the free tier?",
        "Is there a student or educational discount available?",
        "How does your data backup system work?",
        "Do you offer a free trial before committing to a plan?",
        "What integrations are available with other tools?",
        "How can I contact the sales team for a demo?",
        "What is your data privacy and security policy?",
        "Are there any upcoming features on the product roadmap?",
        "How do I provide feedback or suggest a feature?",
        "What are your customer support operating hours?",
        "Is there a mobile application available for download?",
        "Can I use the service for my non-profit organization?",
        "What are the system requirements to run the software?",
        "Do you have documentation or tutorials for beginners?",
        "What is the maximum file upload size allowed?",
        "How does the referral program work?",
        "Is there an API available for custom integrations?",
        "What languages does the platform support?",
        "Can I export all my data if I decide to leave?",
        "Do you provide onboarding support for new customers?",
        "What is the uptime guarantee for your service?",
        "How do I join the community forum?",
    ],
}

# ─── Priority keywords ──────────────────────────────────────────────────────

HIGH_PRIORITY_KEYWORDS = [
    "urgent", "immediately", "critical", "failed", "not working",
    "error", "crashed", "broken", "lost", "hacked", "unauthorized",
    "locked out", "suspended", "fatal",
]

# ─── Augmentation helpers ────────────────────────────────────────────────────

URGENCY_PREFIXES = [
    "URGENT: ", "Please help immediately! ", "This is critical — ",
    "I need this resolved ASAP. ", "Emergency! ",
]

POLITE_SUFFIXES = [
    " Thanks!", " Thank you for your help.", " Appreciate your support.",
    " Looking forward to your response.", " Please advise.",
    " Could you help?", "",
]


def _augment_text(text: str, add_urgency: bool = False) -> str:
    """Add slight variation to a template sentence."""
    t = text
    if add_urgency and random.random() < 0.6:
        t = random.choice(URGENCY_PREFIXES) + t
    t += random.choice(POLITE_SUFFIXES)
    return t


def _assign_priority(text: str) -> str:
    """Rule-based priority tagging using keyword matching."""
    lower = text.lower()
    for kw in HIGH_PRIORITY_KEYWORDS:
        if kw in lower:
            return "High"
    # Heuristic: Technical & Account issues default Medium; rest Low
    return "Medium"


def generate_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic customer-support ticket dataset.

    Parameters
    ----------
    n_samples : int
        Total number of tickets to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns: ticket_id, text, category, priority
    """
    random.seed(seed)
    np.random.seed(seed)

    categories = list(TEMPLATES.keys())
    records = []

    for i in range(n_samples):
        cat = random.choice(categories)
        base = random.choice(TEMPLATES[cat])
        add_urgency = random.random() < 0.25  # 25 % chance
        text = _augment_text(base, add_urgency=add_urgency)
        priority = _assign_priority(text)
        # Override: General Inquiries without urgency → Low
        if cat == "General Inquiry" and priority != "High":
            priority = "Low"
        records.append({
            "ticket_id": f"TKT-{i+1:04d}",
            "text": text,
            "category": cat,
            "priority": priority,
        })

    return pd.DataFrame(records)


def save_dataset(df: pd.DataFrame, path: str = "data/tickets.csv") -> str:
    """Save the dataset to CSV and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    df = generate_dataset(1000)
    p = save_dataset(df)
    print(f"✅ Dataset saved to {p}  ({len(df)} rows)")
    print(df.head(10).to_string(index=False))
