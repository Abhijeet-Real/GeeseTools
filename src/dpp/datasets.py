import os
import pandas as pd

# Get the absolute path of the `data/` folder inside the installed package
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_heart_dataset() -> pd.DataFrame:
    """
    Load the heart.csv dataset from the installed package.
    """
    heart_csv_path = os.path.join(DATA_DIR, "heart.csv")

    if not os.path.exists(heart_csv_path):
        raise FileNotFoundError(f"Dataset not found: {heart_csv_path}")

    return pd.read_csv(heart_csv_path)
