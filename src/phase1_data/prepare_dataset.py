def split_features_target(df):
    """
    Splits dataframe into features (X) and target (y)
    """
    
    if "default" not in df.columns:
        raise Exception("Target column 'default' not found in dataframe")
    
    X = df.drop("default", axis=1)
    
    y = df["default"]
    
    return X, y