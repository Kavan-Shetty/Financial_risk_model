import pandas as pd

def temporal_split(df: pd.DataFrame, split_ratio: float = 0.8):
    
    print("\nPerforming temporal split...")
    
    # Sort by Time column (critical)
    df = df.sort_values("Time")
    
    split_index = int(len(df) * split_ratio)
    
    train_df = df.iloc[:split_index]
    
    test_df = df.iloc[split_index:]
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df