import shap
import pandas as pd

print("shap_explainer module loaded")


def generate_shap_explanations(model, X_train, X_test, sample_size=1000):

    print("\nGenerating SHAP explanations...")

    # sample test set for speed
    if len(X_test) > sample_size:
        X_sample = X_test.sample(sample_size, random_state=42)
    else:
        X_sample = X_test.copy()

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    print("SHAP explanations generated")

    return explainer, shap_values, X_sample


def get_feature_importance(shap_values, X):

    print("\nComputing feature importance...")

    importance = abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    print("Feature importance computed")

    return importance_df