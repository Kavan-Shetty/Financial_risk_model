# src/phase3_model/model_builder.py

from lightgbm import LGBMClassifier


def build_model(scale_pos_weight):
    """
    Build optimized LightGBM model for financial risk prediction.
    Properly regularized to handle extreme imbalance.
    """

    print("\nBuilding optimized LightGBM financial risk model...")

    model = LGBMClassifier(

        objective='binary',
        boosting_type='gbdt',

        # Core learning parameters
        n_estimators=500,
        learning_rate=0.05,

        # Tree complexity control
        num_leaves=31,
        max_depth=6,

        # Prevent overfitting
        min_data_in_leaf=50,

        # Regularization (VERY IMPORTANT)
        lambda_l1=1.0,
        lambda_l2=1.0,

        # Sampling
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,

        # Imbalance handling
        scale_pos_weight=scale_pos_weight,

        # Performance
        random_state=42,
        n_jobs=-1,
        verbosity=-1

    )

    return model