import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize target column name
        if "Class" in df.columns:
            df = df.rename(columns={"Class": "default"})
        
        print(f"Data loaded successfully: {df.shape}")
        
        return df
    
    except Exception as e:
        
        raise Exception(f"Error loading data: {e}")