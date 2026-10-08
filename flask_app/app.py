from flask import Flask, request, render_template
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import mlflow
import dagshub
import pickle
import os
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Set up DagsHub credentials for MLflow tracking
load_dotenv()
dagshub_url = "https://dagshub.com"
dagshub_token = os.getenv("DAGSHUB_TOKEN")
repo_owner = os.getenv("repo_owner")
repo_name = os.getenv('repo_name')

if not all([dagshub_token, repo_owner, repo_name]):
    raise EnvironmentError("Error loading Dagshub credentials.")

os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow_tracking_uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"

mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

# initializing flask app
app = Flask(__name__)

# creating a custom registry and defining custom metrics
registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    name="app_request_count", documentation="Total number of requests to the app", labelnames=["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    name="app_request_latency_seconds", documentation="Latency of requests in seconds", labelnames=["endpoint"], registry=registry
)

# fetching the model from the registry
model_name = os.getenv("model_name")


def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(
        model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(
            model_name, stages=["None"])
    return latest_version[0].version if latest_version else None


model_version = get_latest_model_version(model_name=model_name)

model_uri = f"models:/{model_name}/{model_version}"
print(mlflow.pyfunc.get_model_dependencies(model_uri))

print(f"Fetching model from {model_uri}")
model = mlflow.pyfunc.load_model(model_uri=model_uri)

# Routes


@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time()-start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    data = request.get_json()
    data_df = pd.read_json(data)

    prediction = model.predict(data_df)[0]

    # measuring latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time()-start_time)

    return render_template("index.html", result=prediction)


@app.route("/metrics", methods=['GET'])
def metrics():
    """ Exposing custom prometheus metrics"""
    return generate_latest(registry), 200, {"content-type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(debug=True)
