# src/phase3_model/imbalance_handler.py

import numpy as np


def compute_class_weight(y):
    """
    Compute and cap class imbalance weight for financial risk modelling.
    Prevents model collapse due to extreme imbalance.
    """

    print("\nComputing class imbalance weight...")

    y = np.array(y)

    negative = np.sum(y == 0)
    positive = np.sum(y == 1)

    if positive == 0:
        raise Exception("No positive samples found")

    raw_weight = negative / positive

    # Cap weight to prevent overcompensation
    capped_weight = min(raw_weight, 100)

    print(f"Negative samples: {negative}")
    print(f"Positive samples: {positive}")
    print(f"Raw imbalance weight: {raw_weight:.2f}")
    print(f"Capped imbalance weight used: {capped_weight}")

    return capped_weight