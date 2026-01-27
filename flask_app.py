from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.models.train_model import main as train_main

MODEL_PATH = Path("model.joblib")

app = Flask(__name__)


def load_or_train_model():
    if not MODEL_PATH.exists():
        # Train and save model if not present
        train_main()
    model = joblib.load(MODEL_PATH)
    # Expect the training script to attach feature_cols
    feature_cols = getattr(model, "feature_cols", None)
    return model, feature_cols


model, feature_cols = load_or_train_model()


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Salary prediction API is running",
        "expected_features": feature_cols,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload: Dict[str, Any] = request.get_json(force=True)
        if feature_cols:
            # Validate required features
            missing = [f for f in feature_cols if f not in payload]
            if missing:
                return jsonify({"error": f"Missing features: {missing}"}), 400
        # Construct a single-row DataFrame with expected columns order
        data = {f: payload.get(f) for f in feature_cols} if feature_cols else payload
        X = pd.DataFrame([data])
        y_pred = model.predict(X)
        return jsonify({"prediction": float(y_pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
