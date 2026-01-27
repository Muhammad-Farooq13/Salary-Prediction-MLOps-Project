from pathlib import Path
from typing import Tuple
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_PATH = Path("data/raw/Salary_Data.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_PATH = PROCESSED_DIR / "processed.csv"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing: dropping NA rows and coercing dtypes")
    df = df.dropna()
    # Coerce numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            # leave non-numeric as is
            pass
    return df


def save_processed(df: pd.DataFrame, out_path: Path = PROCESSED_PATH) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved processed data to {out_path}")
    return out_path


def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_raw = load_raw_data()
    df_processed = preprocess(df_raw)
    save_processed(df_processed)
    return df_raw, df_processed


if __name__ == "__main__":
    main()
