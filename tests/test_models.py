from pathlib import Path

from src.models.train_model import train


def test_train_returns_metrics():
    model, metrics = train(Path("data/raw/Salary_Data.csv"))
    assert hasattr(model, "predict")
    for key in ["mae", "mse", "rmse", "r2", "best_alpha"]:
        assert key in metrics
