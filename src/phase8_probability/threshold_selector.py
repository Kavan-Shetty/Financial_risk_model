import numpy as np
from sklearn.metrics import roc_curve

print("threshold_selector module loaded")


def find_optimal_threshold(y_true, probabilities):

    print("\nFinding optimal threshold using KS Statistic...")

    fpr, tpr, thresholds = roc_curve(y_true, probabilities)

    ks_values = tpr - fpr

    optimal_index = np.argmax(ks_values)

    optimal_threshold = thresholds[optimal_index]

    print(f"Optimal threshold found: {optimal_threshold:.6f}")

    return optimal_threshold


def classify_with_threshold(probabilities, threshold):

    print("\nClassifying using optimal threshold...")

    predictions = (probabilities >= threshold).astype(int)

    print("Classification complete")

    return predictions