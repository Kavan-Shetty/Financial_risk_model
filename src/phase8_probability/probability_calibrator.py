import numpy as np
from sklearn.calibration import CalibratedClassifierCV

print("probability_calibrator module loaded")

def calibrate_model(model, X_train, y_train):

    print("\nCalibrating model probabilities...")

    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",   # FIXED
        cv=3
    )

    calibrated_model.fit(X_train, y_train)

    print("Calibration complete")

    return calibrated_model


def get_calibrated_probabilities(model, X):

    probabilities = model.predict_proba(X)[:, 1]

    return probabilities