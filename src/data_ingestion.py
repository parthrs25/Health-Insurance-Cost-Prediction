import os
import logging
import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Public raw URL for Kaggle Medical Cost Personal Dataset
DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

def load_params(config_path: str = "params.yaml") -> dict:
    """Load configuration parameters cleanly with fallback for notebook execution."""
    if not os.path.exists(config_path) and os.path.exists(os.path.join("..", config_path)):
        config_path = os.path.join("..", config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_data(raw_path: str) -> None:
    """Download the dataset from source if not locally present."""
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    if not os.path.exists(raw_path):
        logger.info(f"Downloading raw dataset from {DATA_URL}...")
        response = requests.get(DATA_URL, timeout=15)
        response.raise_for_status()
        with open(raw_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Dataset successfully saved to {raw_path}")
    else:
        logger.info(f"Dataset already exists at {raw_path}")

def load_raw_data(config_path: str = "params.yaml") -> pd.DataFrame:
    """Ingest and return raw dataset."""
    params = load_params(config_path)
    raw_path = params["data"]["raw_path"]
    download_data(raw_path)
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print("\n--- DATASET PREVIEW ---")
    print(df.head())
    print("\n--- DATASET INFO ---")
    print(df.info())
