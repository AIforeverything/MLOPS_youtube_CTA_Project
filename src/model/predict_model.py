import pandas as pd
import numpy as np
import json
from src.utils.load_model import load_model
from src.utils.save_metrics import save_metrics
from src.utils.yaml_loader import yaml_loader
from src.logger.logger import configure_logger
import dagshub
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = configure_logger()

# setting up dagshub for mlflow
load_dotenv()
dagshub_token = os.getenv("DAGSHUB_TOKEN")
dagshub_url = "https://dagshub.com"
repo_owner = os.getenv('repo_owner')
repo_name = os.getenv('repo_name')
if not dagshub_token:
    raise EnvironmentError(
        "dagshub_token  not loaded properly.")
if not dagshub_url:
    raise EnvironmentError(
        "dagshub_url  not loaded properly.")
if not repo_owner:
    raise EnvironmentError(
        "repo_owner  not loaded properly.")
if not repo_name:
    raise EnvironmentError(
        "repo_name  not loaded properly.")


def model_predict(test_data_path: str, model_pipeline_path: str, model_name: str):
    try:
        logger.info("Test Data Loading is started.")
        test_data_df = pd.read_csv(test_data_path)
        if not test_data_df.empty:
            logger.info("Test Data is loaded successfully!")

        X_test = test_data_df.drop(columns=["view_count"])
        y_test = np.log1p(test_data_df["view_count"])

        model = load_model(model_pipeline_path)
        if model:
            logger.info("Model pipeline is loaded successfully!")

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "r2_score": r2
        }
        logger.info('Model evaluation metrics calculated')

        return model, metrics

    except FileNotFoundError as e:
        logger.exception(f"File doesn't exist error")
        raise

    except pd.errors.ParserError as e:
        logger.exception("Data loading error")
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.info('Model info saved to %s', file_path)
    except Exception:
        logger.exception("Error occurred while saving the model")
        raise


def main():
    test_data_path = "./data/processed/test.csv"
    model_pipeline_path = "./models/model.pkl"
    save_metrics_path = './reports/metrics.json'
    model_name = yaml_loader("./params.yaml")["model"]

    try:
        # loading the model, calculating and saving the metrics using below function
        model, metrics = model_predict(
            test_data_path, model_pipeline_path, model_name)
        save_metrics(metrics, save_metrics_path)

        # setting mlflow
        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri(
            f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
        logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        mlflow.set_experiment("MLOPS_youtube_Project")
        with mlflow.start_run() as run:

            # logging the metrics to mlflow
            mlflow.log_metrics(metrics=metrics)

            # logging parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # logging model to mlflow
            mlflow.sklearn.log_model(model, name=f'youtube_model_{model_name}',
                                     skops_trusted_types=["numpy.dtype", 'xgboost.core.Booster', 'xgboost.sklearn.XGBRegressor'])

            # saving model information
            save_model_info(
                run.info.run_id, f'youtube_model_{model_name}', 'reports/experiment_info.json')

            # logging metrics file to mlflow
            mlflow.log_artifact('reports/metrics.json')

    except Exception:
        logger.exception(
            "Error occurred")
        raise


if __name__ == "__main__":
    main()
