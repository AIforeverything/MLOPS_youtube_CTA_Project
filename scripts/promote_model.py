import os
import mlflow
from dotenv import load_dotenv
from src.logger.logger import configure_logger


def promote_model():
    # setting up dagshub credentials for mlflow tracking
    # =========================================================
    # Load environment variables
    # =========================================================
    load_dotenv()
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    repo_owner = os.getenv("REPO_OWNER")
    repo_name = os.getenv("REPO_NAME")
    model_name = os.getenv("MODEL_NAME")

    if not all([dagshub_token, repo_owner, repo_name]):
        raise EnvironmentError(
            "DagsHub environment variables are not loaded properly.")

    # =========================================================
    # Configure MLflow
    # =========================================================

    dagshub_url = "https://dagshub.com"

    mlflow_tracking_uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"

    os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_registry_uri(mlflow_tracking_uri)

    logger = configure_logger()

    logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    logger.info(f"MLflow Registry URI: {mlflow.get_registry_uri()}")

    client = mlflow.MlflowClient()
    latest_version_staging = client.get_latest_versions(
        model_name, stages=["Staging"])[0].version
    logger.info(f"latest_version_staging: {latest_version_staging}")

    # Archieve the current production model
    prod_versions = client.get_latest_versions(
        model_name, stages=["Production"])
    logger.info(f"prod_versions: {prod_versions}")
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage='Archieved'
        )

    # promote the new model to production
    client.transition_model_version_stage(
        model_name=model_name,
        version=latest_version_staging,
        stage='Production',
    )

    print(f"Model versions {latest_version_staging} promoted to production.")


if __name__ == "__main__":
    promote_model()
