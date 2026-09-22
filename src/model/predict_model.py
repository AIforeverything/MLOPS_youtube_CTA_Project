import pandas as pd
import numpy as np
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
if not dagshub_token:
    raise EnvironmentError("dagshub Environment variable is not loaded.")


def model_predict_mlflow(test_data_path: str, model_pipeline_path: str, model_name: str,
                         save_metrics_path: str):
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
            "model": "RandomForestRegressor",
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "r2_score": r2
        }
        save_metrics(metrics, save_metrics_path)
        logger.info('Model evaluation metrics calculated')

        return metrics

    except FileNotFoundError as e:
        logger.exception(f"File doesn't exist error")
        raise

    except pd.errors.ParserError as e:
        logger.exception("Data loading error")
        raise


def main():
    test_data_path = "./data/processed/test.csv"
    model_pipeline_path = "./models/model.pkl"
    save_metrics_path = './reports/metrics.json'
    model_name = yaml_loader("./params.yaml")["model"]

    model_predict_mlflow(
        test_data_path, model_pipeline_path, model_name, save_metrics_path)


if __name__ == "__main__":
    main()
