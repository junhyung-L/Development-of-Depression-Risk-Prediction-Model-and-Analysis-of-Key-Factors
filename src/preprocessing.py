import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def load_and_preprocess_data(file_path):
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
    import os
    # Get current file path to find data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'Depression Student Dataset.csv')
    
    if os.path.exists(data_path):
        df, target = load_and_preprocess_data(data_path)
        print("Data preprocessed successfully.")
        print(df.head())
    else:
        print(f"Data file not found at: {data_path}")
