print("model_loader module loaded")

import joblib
import os


def load_model(path="models"):
    """
    Load trained model and feature names
    """

    model_path = os.path.join(path, "risk_model.pkl")
    features_path = os.path.join(path, "feature_names.pkl")

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)

    print("Model loaded successfully")

    return model, feature_names