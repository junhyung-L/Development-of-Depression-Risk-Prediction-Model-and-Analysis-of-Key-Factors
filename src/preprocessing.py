from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

try:
    from .config import DATA_PATH
except ImportError:
    from config import DATA_PATH


def load_and_preprocess_data(file_path: Path | str = DATA_PATH):
    """Load and encode the survey data used by the legacy model."""
    # Load data
    data = pd.read_csv(file_path)
    
    # Copy data for processing
    processed_data = data.copy()
    
    # Define columns
    numerical_cols = ['Age', 'Study Hours']
    ordinal_cols = ['Academic Pressure', 'Study Satisfaction', 'Financial Stress']
    categorical_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']
    target_col = 'Depression'
    
    # Encode Target
    le = LabelEncoder()
    processed_data[target_col] = le.fit_transform(processed_data[target_col])
    
    # Encode Ordinal Columns
    ordinal_encoder = OrdinalEncoder()
    processed_data[ordinal_cols] = ordinal_encoder.fit_transform(processed_data[ordinal_cols])
    
    # Encode Categorical Columns for modeling (Simple Label Encoding for simplicity in this baseline)
    # In a full pipeline, One-Hot Encoding might be better for non-ordinal categorical variables.
    for col in categorical_cols:
        processed_data[col] = le.fit_transform(processed_data[col])
        
    return processed_data, target_col

if __name__ == "__main__":
    if DATA_PATH.exists():
        df, target = load_and_preprocess_data(DATA_PATH)
        print("Data preprocessed successfully.")
        print(df.head())
    else:
        print(f"Data file not found at: {DATA_PATH}")
