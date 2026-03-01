print("train_model module loaded")

import lightgbm as lgb


def train_model(X_train, y_train, class_weight):

    print("\nTraining LightGBM risk model...")

    model = lgb.LGBMClassifier(

        objective="binary",

        n_estimators=1000,
        learning_rate=0.02,

        num_leaves=128,
        max_depth=-1,

        min_child_samples=5,
        min_child_weight=1e-5,

        subsample=0.9,
        colsample_bytree=0.9,

        reg_alpha=0.1,
        reg_lambda=0.1,

        scale_pos_weight=class_weight,

        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Model training completed successfully")

    return model