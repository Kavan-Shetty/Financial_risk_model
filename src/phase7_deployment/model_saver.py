print("model_saver module loaded")

import joblib
import os


def save_model(model, feature_names, path="models"):
    """
    Save trained model and feature names for deployment
    """

    os.makedirs(path, exist_ok=True)

    model_path = os.path.join(path, "risk_model.pkl")
    features_path = os.path.join(path, "feature_names.pkl")

    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)

    print(f"Model saved successfully at: {model_path}")
    print(f"Feature names saved at: {features_path}")
    print("model_saver module loaded")

import joblib
import os


def save_model(model, feature_names, path="models"):
    """
    Saves trained model and feature names
    """

    os.makedirs(path, exist_ok=True)

    model_path = os.path.join(path, "risk_model.pkl")
    feature_path = os.path.join(path, "feature_names.pkl")

    joblib.dump(model, model_path)
    joblib.dump(feature_names, feature_path)

    print("Model saved successfully")