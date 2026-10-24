# import pandas as pd
# import numpy as np
# import json
# from src.utils.load_model import load_model
# from src.utils.save_metrics import save_metrics
# from src.utils.yaml_loader import yaml_loader
# from src.logger.logger import configure_logger
# import dagshub
# import mlflow
# import mlflow.sklearn
# from mlflow.models import infer_signature
# import os
# from dotenv import load_dotenv
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# logger = configure_logger()

# # setting up dagshub for mlflow
# load_dotenv()
# dagshub_token = os.getenv("DAGSHUB_TOKEN")
# dagshub_url = "https://dagshub.com"
# repo_owner = os.getenv("REPO_OWNER")
# repo_name = os.getenv("REPO_NAME")
# if not all([dagshub_token, repo_owner, repo_name]):
#     raise EnvironmentError(
#         "DagsHub environment variables are not loaded properly.")

# mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
# logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


# def model_predict(test_data_path: str, model_pipeline_path: str):
#     try:
#         logger.info("Test Data Loading is started.")
#         test_data_df = pd.read_csv(test_data_path)
#         if not test_data_df.empty:
#             logger.info("Test Data is loaded successfully!")

#         X_test = test_data_df.drop(columns=["view_count"])
#         y_test = np.log1p(test_data_df["view_count"])

#         model = load_model(model_pipeline_path)
#         if model:
#             logger.info("Model pipeline is loaded successfully!")

#         y_pred = model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         mse = mean_squared_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)

#         metrics = {
#             "mean_absolute_error": mae,
#             "mean_squared_error": mse,
#             "r2_score": r2,
#         }
#         logger.info("Model evaluation metrics calculated")

#         return model, metrics

#     except FileNotFoundError as e:
#         logger.exception(f"File doesn't exist error")
#         raise

#     except pd.errors.ParserError as e:
#         logger.exception("Data loading error")
#         raise


# def save_model_info(
#     run_id: str, model_name: str, model_uri: str, file_path: str, url: str
# ) -> None:
#     """Save MLflow model information to a JSON file."""

#     try:
#         model_info = {
#             "run_id": run_id,
#             "model": model_name,
#             "model_uri": model_uri,
#             "url": url,
#         }

#         with open(file_path, "w") as file:
#             json.dump(model_info, file, indent=4)

#         logger.info("Model info saved to %s", file_path)

#     except Exception:
#         logger.exception("Error occurred while saving the model information")
#         raise


# def main():
#     test_data_path = "./data/processed/test.csv"
#     model_pipeline_path = "./models/model.pkl"
#     save_metrics_path = "./reports/metrics.json"
#     model_name = yaml_loader("./params.yaml")["model"]
#     experiment_name = yaml_loader("./params.yaml")["experiment_name"]

#     test_data_df = pd.read_csv(test_data_path)
#     X_test_sample = test_data_df.drop(columns=["view_count"]).sample(1)
#     y_test_sample = np.log1p(test_data_df["view_count"]).sample(1)

#     try:
#         # loading the model, calculating and saving the metrics using below function
#         model, metrics = model_predict(test_data_path, model_pipeline_path)
#         save_metrics(metrics, save_metrics_path)

#         # ---------------------------------------------------------
#         # MLflow experiment
#         # ---------------------------------------------------------

#         mlflow.set_experiment(experiment_name)

#         with mlflow.start_run() as run:

#             # Log metrics
#             mlflow.log_metrics(metrics)

#             # Log parameters
#             if hasattr(model, "get_params"):

#                 params = model.get_params()

#                 for param_name, param_value in params.items():
#                     mlflow.log_param(param_name, param_value)

#             # -----------------------------------------------------
#             # Log and register model
#             # -----------------------------------------------------

#             model_info = mlflow.sklearn.log_model(
#                 model,
#                 name=model_name,
#                 registered_model_name=model_name,
#                 skops_trusted_types=["numpy.dtype"],
#             )

#             logger.info(f"Model logged successfully: {model_info.model_uri}")

#             logger.info(f"Model registered successfully: {model_name}")

#             # -----------------------------------------------------
#             # Save model information
#             # -----------------------------------------------------

#             save_model_info(
#                 run.info.run_id,
#                 model_name,
#                 model_info.model_uri,
#                 "./reports/experiment_info.json",
#                 mlflow.get_tracking_uri(),
#             )

#             # -----------------------------------------------------
#             # Log metrics file
#             # -----------------------------------------------------

#             mlflow.log_artifact("reports/metrics.json")

#     except Exception:
#         logger.exception("Error occurred")
#         raise


# if __name__ == "__main__":
#     main()


"""Evaluate a trained model, log metrics/model to MLflow, and save run metadata.

The model is deliberately NOT registered here. Registration happens exactly once
in src/model/register_model.py, which assigns the new version the `candidate` alias.
"""

from pathlib import Path
import json
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger.logger import configure_logger
from src.utils.load_model import load_model
from src.utils.save_metrics import save_metrics
from src.utils.yaml_loader import yaml_loader

logger = configure_logger()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_mlflow() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("DAGSHUB_TOKEN")
    owner = os.getenv("REPO_OWNER")
    repo = os.getenv("REPO_NAME")
    if not all([token, owner, repo]):
        raise EnvironmentError(
            "Missing DagsHub configuration. Set DAGSHUB_TOKEN, REPO_OWNER, and REPO_NAME."
        )

    tracking_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)


def model_predict(test_data_path: str, model_pipeline_path: str):
    test_path = Path(test_data_path)
    if not test_path.is_absolute():
        test_path = PROJECT_ROOT / test_path
    model_path = Path(model_pipeline_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    test_df = pd.read_csv(test_path)
    if test_df.empty or "view_count" not in test_df.columns:
        raise ValueError(
            f"Test data must be non-empty and contain view_count: {test_path}"
        )

    X_test = test_df.drop(columns=["view_count"])
    # Training uses log1p(view_count), so evaluate on that same target scale.
    y_test = np.log1p(test_df["view_count"])
    model = load_model(str(model_path))
    y_pred = np.asarray(model.predict(X_test)).reshape(-1)

    metrics = {
        "mean_absolute_error": float(mean_absolute_error(y_test, y_pred)),
        "mean_squared_error": float(mean_squared_error(y_test, y_pred)),
        "r2_score": float(r2_score(y_test, y_pred)),
    }
    logger.info("Candidate metrics (log1p target scale): %s", metrics)
    return model, metrics, X_test


def save_model_info(
    run_id: str, model_name: str, model_uri: str, file_path: Path, url: str
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "model": model_name,
        "model_uri": model_uri,
        "url": url,
    }
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved model information to %s", file_path)


def main() -> None:
    configure_mlflow()
    params = yaml_loader(str(PROJECT_ROOT / "params.yaml"))
    model_name = os.getenv("MODEL_NAME") or params["model"]
    experiment_name = params["experiment_name"]

    model, metrics, X_test = model_predict(
        "data/processed/test.csv", "models/model.pkl"
    )
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"
    save_metrics(metrics, str(metrics_path))

    predictions = np.asarray(model.predict(X_test.head(min(5, len(X_test))))).reshape(
        -1
    )
    signature = infer_signature(X_test.head(min(5, len(X_test))), predictions)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"model_role": "candidate", "promotion_metric": "r2_score"})
        if hasattr(model, "get_params"):
            for key, value in model.get_params(deep=True).items():
                if value is None or isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, str(value))

        # Important: do not pass registered_model_name here; register_model.py
        # registers this run's model once and assigns the candidate alias.
        input_example = X_test.head(1).copy()
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=["numpy.dtype"],
        )

        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        save_model_info(
            run_id=run.info.run_id,
            model_name=model_name,
            model_uri=model_info.model_uri,
            file_path=reports_dir / "experiment_info.json",
            url=mlflow.get_tracking_uri(),
        )
        logger.info("Logged candidate model artifact: %s", model_info.model_uri)


if __name__ == "__main__":
    main()
