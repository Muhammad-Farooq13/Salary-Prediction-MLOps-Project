from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.load_data import load_raw_data, preprocess
from src.features.build_features import get_feature_target_columns, build_preprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = Path("model.joblib")


def train(data_path: Path = Path("data/raw/Salary_Data.csv")) -> Tuple[Pipeline, Dict[str, float]]:
    df = load_raw_data(data_path)
    df = preprocess(df)

    feature_cols, target_col = get_feature_target_columns(df, target="Salary")
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(feature_cols)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge())
    ])

    param_grid = {
        "model__alpha": [0.1, 1.0, 10.0, 100.0],
    }

    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model: Pipeline = grid.best_estimator_
    # Attach expected feature columns for inference
    setattr(best_model, "feature_cols", feature_cols)

    y_pred = best_model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "r2": float(r2_score(y_test, y_pred)),
        "best_alpha": float(grid.best_params_["model__alpha"]),
    }

    logger.info(f"Training complete. Metrics: {metrics}")
    return best_model, metrics


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> Path:
    joblib.dump(model, path)
    logger.info(f"Saved model pipeline to {path}")
    return path


def main() -> Dict[str, float]:
    model, metrics = train()
    save_model(model, MODEL_PATH)
    return metrics


if __name__ == "__main__":
    main()
