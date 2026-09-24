from src.logger.logger import configure_logger
from src.utils.json_loader import load_json
import mlflow
import os
from dotenv import load_dotenv

import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings("ignore")

logger = configure_logger()

# Set up DagsHub credentials for MLflow tracking
load_dotenv()
dagshub_token = os.getenv("dagshub_token")
dagshub_url = "https://dagshub.com"
repo_owner = os.getenv("repo_owner")
repo_name = os.getenv("repo_name")

if not (dagshub_token or repo_owner or repo_name):
    logger.exception("Environment credentials are not loaded properly.")
    raise EnvironmentError("Environment credentials are not loaded properly.")

os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


def register_model(model_name: str, model_info: dict) -> None:
    """ Register the model to MLFLOW model registry."""
    try:
        model_uri = f"runs:{model_info['run_id']}/{model_info['model']}"

        # registering the model
        model_version = mlflow.register_model(
            model_uri=model_uri, name=model_name)
        # transition/sending the model to 'staging'
        client = mlflow.client.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Staging',
        )

        logger.info(
            f"Model: {model_name},Version:{model_version.version} registered and transitioned to Staging")

    except Exception as e:
        logger.exception("Error during model registration.")
        raise e


def main():
    try:
        model_info_file_path = "./reports/experiment_info.json"
        model_info = load_json(model_info_file_path)
        model_name = model_info["model"]
        register_model(model_name=model_name, model_info=model_info)

    except Exception as e:
        logger.exception("Error during model registration.")
        raise e


if __name__ == "__main__":
    main()
