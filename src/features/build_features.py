from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_feature_target_columns(df: pd.DataFrame, target: str = "Salary") -> Tuple[List[str], str]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    feature_cols = [c for c in df.columns if c != target]
    return feature_cols, target


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    numeric_features = feature_cols
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    logger.info(f"Built preprocessor for features: {numeric_features}")
    return preprocessor
