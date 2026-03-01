print("predict module loaded")

def predict_probabilities(model, X_test):

    print("\nPredicting default probabilities...")

    probabilities = model.predict_proba(X_test)[:, 1]

    print("Prediction complete")

    return probabilities