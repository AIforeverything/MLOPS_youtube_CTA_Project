# from src.logger.logger import configure_logger
# from src.utils.json_loader import load_json
# import mlflow
# import os
# from dotenv import load_dotenv

# import warnings

# warnings.simplefilter("ignore", UserWarning)
# warnings.filterwarnings("ignore")

# logger = configure_logger()

# # Set up DagsHub credentials for MLflow tracking
# load_dotenv()
# dagshub_token = os.getenv("DAGSHUB_TOKEN")
# dagshub_url = "https://dagshub.com"
# repo_owner = os.getenv("REPO_OWNER")
# repo_name = os.getenv("REPO_NAME")

# if not dagshub_token:
#     logger.exception("dagshub_token  not loaded properly.")
#     raise EnvironmentError("dagshub_token  not loaded properly.")
# if not repo_owner:
#     logger.exception("repo_owner  not loaded properly.")
#     raise EnvironmentError("repo_owner  not loaded properly.")
# if not repo_name:
#     logger.exception("repo_name  not loaded properly.")
#     raise EnvironmentError("repo_name  not loaded properly.")

# os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


# def register_model(model_name: str, model_info: dict) -> None:
#     """Register the model to MLFLOW model registry."""
#     try:
#         model_uri = f"runs:/{model_info['run_id']}/{model_info['model']}"

#         # registering the model
#         model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
#         # transition/sending the model to 'staging'
#         client = mlflow.client.MlflowClient()
#         client.transition_model_version_stage(
#             name=model_name,
#             version=model_version.version,
#             stage="Staging",
#         )

#         logger.info(
#             f"Model: {model_name},Version:{model_version.version} registered and transitioned to Staging"
#         )

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
import os
from dotenv import load_dotenv
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

logger = configure_logger()


# ---------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# ---------------------------------------------------------

load_dotenv()

dagshub_token = os.getenv("DAGSHUB_TOKEN")
dagshub_url = "https://dagshub.com"
repo_owner = os.getenv("REPO_OWNER")
repo_name = os.getenv("REPO_NAME")


if not dagshub_token:
    logger.error("dagshub_token not loaded properly.")
    raise EnvironmentError("dagshub_token not loaded properly.")

if not repo_owner:
    logger.error("repo_owner not loaded properly.")
    raise EnvironmentError("repo_owner not loaded properly.")

if not repo_name:
    logger.error("repo_name not loaded properly.")
    raise EnvironmentError("repo_name not loaded properly.")


# DagsHub authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


# IMPORTANT:
# Set DagsHub as the MLflow tracking server
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")


# ---------------------------------------------------------
# Register model
# ---------------------------------------------------------


def register_model(model_name: str, model_info: dict) -> None:
    """Register the model to MLflow model registry."""

    try:

        model_uri = f"runs:/{model_info['run_id']}/{model_info['model']}"

        logger.info(f"Model URI: {model_uri}")

        # Register model
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Transition model to Staging
        client = mlflow.MlflowClient()

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
        )

        logger.info(
            f"Model: {model_name}, "
            f"Version: {model_version.version} "
            f"registered and transitioned to Staging"
        )

    except Exception:
        logger.exception("Error during model registration.")
        raise


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():

    try:

        model_info_file_path = "./reports/experiment_info.json"

        model_info = load_json(model_info_file_path)

        model_name = model_info["model"]

        register_model(model_name=model_name, model_info=model_info)

    except Exception:
        logger.exception("Error during model registration.")
        raise


if __name__ == "__main__":
    main()
