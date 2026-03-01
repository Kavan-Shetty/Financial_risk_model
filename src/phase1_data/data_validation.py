import pandas as pd

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    
    print("\nRunning data validation...")
    
    # Check empty dataframe
    if df.empty:
        raise Exception("Dataset is empty")
    
    # Check missing values
    missing = df.isnull().sum().sum()
    
    print(f"Missing values: {missing}")
    
    # Check duplicates
    duplicates = df.duplicated().sum()
    
    print(f"Duplicate rows: {duplicates}")
    
    if duplicates > 0:
        df = df.drop_duplicates()
        print("Duplicates removed")
    
    # Check target variable
    if "default" not in df.columns:
        raise Exception("Target column 'default' not found")
    
    print("\nTarget distribution:")
    print(df["default"].value_counts(normalize=True))
    
    return df