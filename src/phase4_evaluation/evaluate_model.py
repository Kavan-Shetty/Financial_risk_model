# src/phase4_evaluation/evaluate_model.py

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix


def evaluate_model(y_test, probabilities):

    print("\n=========================")
    print("PHASE 4: MODEL EVALUATION")
    print("=========================")

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"\nROC-AUC Score: {roc_auc:.4f}")


    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    pr_auc = auc(recall, precision)

    print(f"PR-AUC Score: {pr_auc:.4f}")


    # KS Statistic
    ks = compute_ks(y_test, probabilities)

    print(f"KS Statistic: {ks:.4f}")


    # Confusion matrix at threshold
    threshold = 0.5

    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")
    print(cm)


    return roc_auc, pr_auc, ks



def compute_ks(y_true, probabilities):

    data = list(zip(probabilities, y_true))

    data.sort(key=lambda x: x[0])

    total_good = sum(1 - y for _, y in data)
    total_bad = sum(y for _, y in data)

    cum_good = 0
    cum_bad = 0

    ks = 0

    for prob, y in data:

        if y == 1:
            cum_bad += 1
        else:
            cum_good += 1

        ks_value = abs((cum_bad / total_bad) - (cum_good / total_good))

        ks = max(ks, ks_value)

    return ks