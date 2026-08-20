"""Shared paths and model defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Depression Student Dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RANDOM_SEED = 42
TEST_SIZE = 0.20
FEATURES = ["Dietary Habits", "Have you ever had suicidal thoughts ?", "Academic Pressure", "Financial Stress"]
