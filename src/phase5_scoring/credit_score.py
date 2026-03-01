import numpy as np

print("credit_score module loaded")


# =========================
# CREDIT SCORE GENERATION
# =========================

def generate_credit_scores(probabilities):
    """
    Convert fraud probabilities into credit scores (300–850)
    using percentile ranking for proper distribution.
    """

    probabilities = np.array(probabilities)

    # Rank probabilities
    ranks = probabilities.argsort().argsort()

    percentiles = ranks / len(probabilities)

    # Convert to credit score scale
    scores = 850 - (percentiles * 550)

    scores = scores.astype(int)

    return scores


# =========================
# RISK CATEGORY ASSIGNMENT
# =========================

def assign_risk_category(scores):

    categories = []

    for score in scores:

        if score >= 800:
            categories.append("Excellent")

        elif score >= 740:
            categories.append("Very Good")

        elif score >= 670:
            categories.append("Good")

        elif score >= 580:
            categories.append("Fair")

        else:
            categories.append("High Risk")

    return categories


# =========================
# FRAUD LABEL ASSIGNMENT
# =========================

def assign_fraud_label(probabilities, threshold=0.5):

    probabilities = np.array(probabilities)

    labels = np.where(
        probabilities >= threshold,
        "Fraud",
        "Safe"
    )

    return labels