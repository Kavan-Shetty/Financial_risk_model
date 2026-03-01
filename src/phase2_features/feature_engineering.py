import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create financial risk features
    """
    
    print("\nEngineering financial features...")
    
    df = df.copy()
    
    # Log transform Amount (reduces skew)
    df["Amount_log"] = np.log1p(df["Amount"])
    
    # Relative transaction size
    df["Amount_relative"] = df["Amount"] / df["Amount"].mean()
    
    # Time difference between transactions
    df["Time_diff"] = df["Time"].diff().fillna(0)
    
    # Interaction features (important risk indicators)
    df["V14_V17_interaction"] = df["V14"] * df["V17"]
    
    df["V12_V14_interaction"] = df["V12"] * df["V14"]
    
    # Composite risk score
    risk_features = ["V10", "V12", "V14", "V17"]
    
    df["Risk_score"] = df[risk_features].abs().sum(axis=1)
    
    print("Feature engineering complete")
    
    return df