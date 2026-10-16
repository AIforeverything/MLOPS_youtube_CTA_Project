from flask import Flask, request, render_template, jsonify
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
)

import mlflow
import os
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


# =========================================================
# Environment variables and DagsHub configuration
# =========================================================

load_dotenv()

dagshub_url = "https://dagshub.com"
dagshub_token = os.getenv("DAGSHUB_TOKEN", "").strip()
repo_owner = os.getenv("REPO_OWNER", "").strip().strip("'\"")
repo_name = os.getenv("REPO_NAME", "").strip().strip("'\"")
model_name = os.getenv("MODEL_NAME", "").strip().strip("'\"")

required_env_vars = {
    "DAGSHUB_TOKEN": dagshub_token,
    "repo_owner": repo_owner,
    "repo_name": repo_name,
    "model_name": model_name,
}

missing_vars = [key for key, value in required_env_vars.items() if not value]

if missing_vars:
    raise EnvironmentError(
        f"Missing environment variables: {', '.join(missing_vars)}")

os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow_tracking_uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"

mlflow.set_tracking_uri(mlflow_tracking_uri)


# =========================================================
# Flask application and Prometheus metrics
# =========================================================

app = Flask(__name__)

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests to the app",
    ["method", "endpoint"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    registry=registry,
)


# =========================================================
# Load registered MLflow model
# =========================================================


def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()

    # Prefer Production if a Production version exists.
    versions = client.get_latest_versions(
        model_name,
        stages=["Production"],
    )

    # Fall back to an unassigned-stage version.
    if not versions:
        versions = client.get_latest_versions(
            model_name,
            stages=["None"],
        )

    if not versions:
        raise RuntimeError(
            f"No eligible model version found for '{model_name}'.")

    return versions[0].version


print("MLflow tracking URI:", mlflow_tracking_uri)
print("Requested model:", repr(model_name))

model_version = get_latest_model_version(model_name)

model_uri = f"models:/{model_name}/{model_version}"

print(f"Fetching model from {model_uri}")

start_time_model = time.time()

model = mlflow.pyfunc.load_model(model_uri=model_uri)

print(f"Model loaded in " f"{time.time() - start_time_model:.2f} seconds")


# =========================================================
# Routes
# =========================================================


@app.route("/", methods=["GET"])
def home():
    REQUEST_COUNT.labels(
        method="GET",
        endpoint="/",
    ).inc()

    start_time = time.time()

    response = render_template("index.html", result=None)

    REQUEST_LATENCY.labels(
        endpoint="/",
    ).observe(time.time() - start_time)

    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/predict",
    ).inc()

    start_time = time.time()

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return (
            jsonify(
                {"error": "Expected a JSON object containing prediction features."}
            ),
            400,
        )

    required_features = [
        "category",
        "subscriber_count",
        "channel_view_count",
        "duration_seconds",
    ]

    missing = [name for name in required_features if name not in payload]

    if missing:
        return (
            jsonify({"error": f"Missing required features: {', '.join(missing)}"}),
            400,
        )

    # handling missing categories
    required_categories = [
        "Education",
        "Entertainment",
        "Finance",
        "Fitness",
        "Food",
        "Gaming",
        "News",
        "Science",
        "Technology",
        "Travel",
    ]
    if payload["category"] not in required_categories:
        payload["category"] = "Education"

    try:
        input_df = pd.DataFrame(
            [
                {
                    "category": str(payload["category"]),
                    "subscriber_count": float(payload["subscriber_count"]),
                    "channel_view_count": float(payload["channel_view_count"]),
                    "duration_seconds": float(payload["duration_seconds"]),
                }
            ]
        )

        numeric_features = [
            "subscriber_count",
            "channel_view_count",
            "duration_seconds",
        ]

        if not np.isfinite(input_df[numeric_features].to_numpy(dtype=float)).all():
            return (
                jsonify({"error": "Numeric features must contain finite values."}),
                400,
            )

        if (input_df[numeric_features] < 0).any().any():
            return jsonify({"error": "Numeric features cannot be negative."}), 400

        print("Prediction input columns:", input_df.columns.tolist())

        prediction = np.expm1(model.predict(input_df)[0])

        REQUEST_LATENCY.labels(
            endpoint="/predict",
        ).observe(time.time() - start_time)

        return jsonify({"prediction": float(prediction)})

    except (TypeError, ValueError, KeyError) as exc:
        return jsonify({"error": f"Invalid inputs: {exc}"}), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({"error": "Prediction failed. Check Flask server logs."}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    return (
        generate_latest(registry),
        200,
        {"Content-Type": CONTENT_TYPE_LATEST},
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
    )
