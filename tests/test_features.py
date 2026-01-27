import pandas as pd
from pathlib import Path

from src.data.load_data import load_raw_data, preprocess
from src.features.build_features import get_feature_target_columns, build_preprocessor


def test_build_preprocessor_runs():
    df = preprocess(load_raw_data(Path("data/raw/Salary_Data.csv")))
    features, target = get_feature_target_columns(df, target="Salary")
    pre = build_preprocessor(features)
    X = df[features]
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
