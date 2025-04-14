import os
import pandas as pd

# Get the absolute path of the `data/` folder inside the installed package
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_heart_dataset() -> pd.DataFrame:
    """
    Load the heart.csv dataset from the installed package.
    """
    default_dataset_path = os.path.join(DATA_DIR, "heart.csv")

    if not os.path.exists(default_dataset_path):
        raise FileNotFoundError(f"Default dataset not found: {default_dataset_path}")

    return pd.read_csv(default_dataset_path)
