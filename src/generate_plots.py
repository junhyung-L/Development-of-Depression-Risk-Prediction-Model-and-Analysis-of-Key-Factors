from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
try:
    from .config import DATA_PATH, FEATURES, RESULTS_DIR
    from .preprocessing import load_and_preprocess_data
except ImportError:
    from config import DATA_PATH, FEATURES, RESULTS_DIR
    from preprocessing import load_and_preprocess_data

def generate_and_save_plots(data_path: Path = DATA_PATH, images_path: Path | None = None):
    """Generate project figures from the configured survey dataset."""
    images_path = images_path or (RESULTS_DIR / "figures")
    images_path.mkdir(parents=True, exist_ok=True)
        
    # Load and preprocess
    df, target = load_and_preprocess_data(data_path)
    
    # 1. Spearman Correlation Heatmap
    numerical_cols = ['Age', 'Study Hours']
    ordinal_cols = ['Academic Pressure', 'Study Satisfaction', 'Financial Stress']
    cols_of_interest = numerical_cols + ordinal_cols + [target]
    
    correlation_matrix = df[cols_of_interest].corr(method='spearman')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={'shrink': .8})
    plt.title('Spearman Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(images_path / 'spearman_heatmap.png')
    plt.close()
    
    # 2. Correlation with Depression
    depression_corr = correlation_matrix['Depression'].drop('Depression')
    plt.figure(figsize=(10, 6))
    sns.barplot(y=depression_corr.index, x=depression_corr.values, palette='viridis')
    plt.title('Correlation with Depression')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(0, color='grey', linestyle='--')
    plt.tight_layout()
    plt.savefig(images_path / 'correlation_with_depression.png')
    plt.close()
    
    # 3. Countplot: Dietary Habits
    # Need original labels for countplots to make sense
    raw_data = pd.read_csv(data_path)
    plt.figure(figsize=(8, 6))
    sns.countplot(data=raw_data, x='Dietary Habits', hue='Depression', palette='Set2')
    plt.title('Depression by Dietary Habits')
    plt.tight_layout()
    plt.savefig(images_path / 'countplot_dietary_habits.png')
    plt.close()
    
    # 4. Countplot: Suicidal Thoughts
    plt.figure(figsize=(8, 6))
    sns.countplot(data=raw_data, x='Have you ever had suicidal thoughts ?', hue='Depression', palette='Set1')
    plt.title('Depression by Suicidal Thoughts')
    plt.tight_layout()
    plt.savefig(images_path / 'countplot_suicidal_thoughts.png')
    plt.close()
    
    # 5. Learning Curve
    features = FEATURES
    X = df[features]
    y = df[target]
    
    model = LogisticRegression()
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Training accuracy', marker='o')
    plt.plot(train_sizes, test_mean, label='Cross-validation accuracy', marker='s')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Learning Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(images_path / 'learning_curve.png')
    plt.close()
    
    print("All plots generated and saved successfully in 'images/' folder.")

if __name__ == "__main__":
    generate_and_save_plots()
