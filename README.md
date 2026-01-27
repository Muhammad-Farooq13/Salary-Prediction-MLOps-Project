# Salary Prediction Project

[![CI](https://github.com/Muhammad-Farooq-13/salary/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq-13/salary/actions/workflows/ci.yml)

A production-ready data science project that predicts salary from years of experience, built with thoughtful MLOps practices, unit testing, and deployment via Flask or Docker.

## Objective
Model the relationship between `YearsExperience` and `Salary`, deliver a reproducible pipeline for data processing, model training, evaluation, and provide a REST API for predictions.

## Maintainer
- Name: Muhammad Farooq
- Email: mfarooqshafee333@gmail.com
- GitHub: https://github.com/Muhammad-Farooq-13

## Methodology
- Data ingestion and preprocessing with `pandas`.
- Feature scaling via `StandardScaler`.
- Model training using `Ridge` regression and `GridSearchCV` for hyperparameter tuning.
- Evaluation metrics: MAE, MSE, RMSE, and R².
- MLOps: version control (Git), unit tests (pytest), automated pipeline (`mlops_pipeline.py`), experiment tracking (MLflow), and containerized deployment (Docker + Gunicorn).

## Dataset Overview
- Source: bundled file `data/raw/Salary_Data.csv`.
- Typical columns: `YearsExperience` (numeric), `Salary` (numeric).
- Preprocessing: drop missing values, coerce numeric types.

## Project Structure
```
.
├── data
│   ├── raw/Salary_Data.csv
│   └── processed/processed.csv
├── notebooks
│   ├── exploration.ipynb
│   └── hyperparameter_tuning.ipynb
├── src
│   ├── data/load_data.py
│   ├── features/build_features.py
│   ├── models/train_model.py
│   ├── visualization/plots.py
│   └── utils/logger.py
├── tests
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── flask_app.py
├── mlops_pipeline.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quickstart (Local)
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies.
3. Run unit tests.
4. Train the model.
5. Start the API.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.models.train_model
python flask_app.py
```

- API endpoints:
  - `GET /` – health with expected feature list
  - `POST /predict` – JSON payload example:

```json
{
  "YearsExperience": 2.0
}
```

Response:
```json
{
  "prediction": 45678.12
}
```

## Docker Deployment
Build the image and run the container:
```bash
docker build -t salary-app .
docker run -p 5000:5000 salary-app
```

Test the API:
```bash
curl http://localhost:5000/
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"YearsExperience": 2.0}'
```

## MLOps Integration
- **Version Control**: Commit code, notebooks, and config to GitHub.
- **Automated Testing**: `pytest` covers data loading, feature building, and model training.
- **Pipeline Script**: `mlops_pipeline.py` runs tests, trains, and logs metrics to MLflow.
- **Experiment Tracking**: MLflow logs parameters, metrics, and artifacts locally.
- **Reproducibility**: `requirements.txt` ensures consistent environments.

### Continuous Integration (GitHub Actions)
A ready-to-use workflow is provided at [.github/workflows/ci.yml](.github/workflows/ci.yml). It:
- Checks out the repo and sets up Python 3.11
- Installs dependencies
- Runs unit tests
- Executes `mlops_pipeline.py` to train and log metrics
- Uploads `model.joblib` and MLflow run artifacts

Enable by pushing this repository to GitHub; the workflow runs on pushes and pull requests to `main`/`master`.

### Docker Compose + MLflow Tracking
Run both the API and an MLflow tracking server:
```bash
docker compose up -d
```
- API: http://localhost:5000
- MLflow UI: http://localhost:5001

The app is configured (via Compose) to use MLflow at `http://mlflow:5001` for experiment tracking.
If you fork this repo or use a different repository name, update the CI badge path accordingly.

To run the pipeline locally:
```bash
python mlops_pipeline.py
```

## Model Development
- **Selection**: Ridge regression chosen for simplicity and regularization; alternatives (Lasso, LinearRegression) can be added.
- **Evaluation**: MAE, MSE, RMSE, R² reported; cross-validation via grid search.
- **Hyperparameter Tuning**: Scripts use `GridSearchCV`; see `notebooks/hyperparameter_tuning.ipynb` for interactive exploration.

## Testing
Run all tests:
```bash
python -m pytest -q
```

Tests include:
- Data loading and preprocessing success.
- Feature pipeline construction.
- Training produces a valid model and metrics.

## Notes
- If `model.joblib` is missing, the Flask app will train a model on startup.
- Update target/feature column names if your dataset schema differs.
- For CI, integrate `mlops_pipeline.py` in your GitHub Actions workflow.
