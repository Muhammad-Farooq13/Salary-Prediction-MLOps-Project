"""
Simple MLOps pipeline script for CI/CD hooks.
- Runs unit tests
- Trains model and saves artifact
- Logs params and metrics to MLflow (local tracking)
"""
from pathlib import Path
import subprocess
import sys

import mlflow

from src.models.train_model import train, save_model

MODEL_PATH = Path("model.joblib")


def run_tests() -> None:
    print("Running unit tests...")
    subprocess.check_call([sys.executable, "-m", "pytest", "-q"])  # quiet mode


def run_training_and_tracking() -> None:
    print("Training model and logging to MLflow...")
    with mlflow.start_run(run_name="salary_regression"):
        model, metrics = train()
        mlflow.log_params({"model": "Ridge"})
        mlflow.log_metrics(metrics)
        path = save_model(model, MODEL_PATH)
        mlflow.log_artifact(str(path))
        print("Metrics:", metrics)


def main() -> None:
    run_tests()
    run_training_and_tracking()


if __name__ == "__main__":
    main()
