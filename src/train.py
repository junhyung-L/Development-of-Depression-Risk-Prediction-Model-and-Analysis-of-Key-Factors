from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import statsmodels.api as sm
try:
    from .config import DATA_PATH, FEATURES, RANDOM_SEED, RESULTS_DIR, TEST_SIZE
    from .preprocessing import load_and_preprocess_data
except ImportError:
    from config import DATA_PATH, FEATURES, RANDOM_SEED, RESULTS_DIR, TEST_SIZE
    from preprocessing import load_and_preprocess_data

def train_model(data_path: Path = DATA_PATH, results_path: Path = RESULTS_DIR):
    """Fit the legacy four-feature logistic model and save its text report."""
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess
    df, target = load_and_preprocess_data(data_path)
    
    # Selected features based on statistical significance analysis in notebook
    # 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Academic Pressure', 'Financial Stress'
    features = FEATURES
    
    X = df[features]
    y = df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    # Train Logistic Regression (Sklearn)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    
    # Statsmodels for detailed summary (p-values)
    X_with_constant = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_with_constant)
    result = logit_model.fit()
    
    print("\n" + "="*50)
    print("Statsmodels Logistic Regression Summary")
    print("="*50)
    print(result.summary())
    
    # Save results summary to results folder
    with open(results_path / 'model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Statsmodels Summary:\n")
        f.write(str(result.summary()))

if __name__ == "__main__":
    train_model()
