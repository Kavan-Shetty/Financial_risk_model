import pandas as pd


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate engineered features for integrity and safety
    """
    
    print("\nValidating engineered features...")
    
    # Check dataframe is not empty
    if df.empty:
        raise Exception("Feature dataframe is empty")
    
    
    # Check for null values
    null_count = df.isnull().sum().sum()
    
    if null_count > 0:
        raise Exception(f"Feature engineering introduced {null_count} null values")
    
    
    # Check for infinite values
    import numpy as np
    
    if np.isinf(df.values).any():
        raise Exception("Infinite values detected in features")
    
    
    print("Feature validation passed")
    
    return df