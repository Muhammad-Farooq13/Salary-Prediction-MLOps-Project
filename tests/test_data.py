import pandas as pd
from pathlib import Path

from src.data.load_data import load_raw_data, preprocess


def test_load_raw_data_exists():
    df = load_raw_data(Path("data/raw/Salary_Data.csv"))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_preprocess_drops_na_and_numeric():
    df = load_raw_data(Path("data/raw/Salary_Data.csv"))
    df2 = preprocess(df)
    assert len(df2) <= len(df)
    # Expect numeric columns to be coercible
    assert df2.select_dtypes(include="number").shape[1] >= 1
