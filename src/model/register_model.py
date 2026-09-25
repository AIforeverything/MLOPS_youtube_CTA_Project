# from src.logger.logger import configure_logger
# from src.utils.json_loader import load_json
# import mlflow
# import os
# from dotenv import load_dotenv

# import warnings
# warnings.simplefilter('ignore', UserWarning)
# warnings.filterwarnings("ignore")

# logger = configure_logger()

# # Set up DagsHub credentials for MLflow tracking
# load_dotenv()
# dagshub_token = os.getenv("dagshub_token")
# dagshub_url = "https://dagshub.com"
# repo_owner = os.getenv("repo_owner")
# repo_name = os.getenv("repo_name")

# if not all(dagshub_token or repo_owner or repo_name):
#     logger.exception("Environment credentials are not loaded properly.")
#     raise EnvironmentError("Environment credentials are not loaded properly.")

# mlflow_tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

# mlflow.set_tracking_uri(mlflow_tracking_uri)
# mlflow.set_registry_uri(mlflow_tracking_uri)
# os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


# def register_model(model_name: str, model_info: dict) -> None:
#     """ Register the model to MLFLOW model registry."""
#     try:
#         model_uri = f"runs:/{model_info['run_id']}/{model_info['model']}"

#         # registering the model
#         model_version = mlflow.register_model(
#             model_uri=model_uri, name=model_name)
#         # transition/sending the model to 'staging'
#         client = mlflow.client.MlflowClient()
#         client.transition_model_version_stage(
#             name=model_name,
#             version=model_version.version,
#             stage='Staging',
#         )

#         logger.info(
#             f"Model: {model_name},Version:{model_version.version} registered and transitioned to Staging")

#     except Exception as e:
#         logger.exception("Error during model registration.")
#         raise e


# def main():
#     try:
#         model_info_file_path = "./reports/experiment_info.json"
#         model_info = load_json(model_info_file_path)
#         model_name = model_info["model"]
#         register_model(model_name=model_name, model_info=model_info)

#     except Exception as e:
#         logger.exception("Error during model registration.")
#         raise e


# if __name__ == "__main__":
#     main()

from src.logger.logger import configure_logger
from src.utils.json_loader import load_json

import mlflow
from mlflow.tracking import MlflowClient

import os
from dotenv import load_dotenv

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

logger = configure_logger()


# =========================================================
# Load environment variables
# =========================================================

load_dotenv()

dagshub_token = os.getenv("DAGSHUB_TOKEN")
repo_owner = os.getenv("repo_owner")
repo_name = os.getenv("repo_name")

if not all([dagshub_token, repo_owner, repo_name]):
    raise EnvironmentError(
        "DagsHub environment variables are not loaded properly."
    )


# =========================================================
# Configure MLflow
# =========================================================

dagshub_url = "https://dagshub.com"

mlflow_tracking_uri = (
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)

os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(
    mlflow_tracking_uri
)

mlflow.set_registry_uri(
    mlflow_tracking_uri
)

logger.info(
    f"MLflow Tracking URI: {mlflow.get_tracking_uri()}"
)

logger.info(
    f"MLflow Registry URI: {mlflow.get_registry_uri()}"
)


# =========================================================
# Register model
# =========================================================

def register_model(
    model_name: str,
    model_info: dict
) -> None:

    try:

        run_id = model_info["run_id"]

        # -------------------------------------------------
        # IMPORTANT:
        # Use model_uri saved by predict_model.py
        # -------------------------------------------------

        model_uri = model_info.get("model_uri")

        if not model_uri:
            raise ValueError(
                "model_uri is missing from experiment_info.json"
            )

        logger.info(
            f"Run ID: {run_id}"
        )

        logger.info(
            f"Model URI: {model_uri}"
        )

        # -------------------------------------------------
        # Verify run exists on DagsHub
        # -------------------------------------------------

        client = MlflowClient()

        run = client.get_run(run_id)

        logger.info(
            f"Run found successfully: {run.info.run_id}"
        )

        # -------------------------------------------------
        # Register model
        # -------------------------------------------------

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logger.info(
            f"Model registered successfully: "
            f"{model_name}, "
            f"Version: {model_version.version}"
        )

    except Exception:
        logger.exception(
            "Error during model registration."
        )
        raise


# =========================================================
# Main
# =========================================================

def main():

    try:

        model_info_file_path = (
            "./reports/experiment_info.json"
        )

        model_info = load_json(
            model_info_file_path
        )

        model_name = model_info["model"]

        register_model(
            model_name=model_name,
            model_info=model_info
        )

    except Exception:
        logger.exception(
            "Error during model registration."
        )
        raise


if __name__ == "__main__":
    main()
