# load test + signature test + performance test
import unittest
import mlflow
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
from src.logger.logger import configure_logger
logger = configure_logger()


class TestModelLoading(unittest.TestCase):
    # testing model loading from dagshub
    load_dotenv()

    @classmethod
    def setUpClass(cls):
        # setup dagshub credentials for model loading
        DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
        REPO_OWNER = os.getenv("REPO_OWNER")
        REPO_NAME = os.getenv("REPO_NAME")
        MODEL_NAME = os.getenv("REPO_NAME")

        if not all([DAGSHUB_TOKEN, REPO_OWNER, REPO_NAME, MODEL_NAME]):
            logger.exception(
                "Error while loading env variables : DAGSHUB_TOKEN, REPO_OWNER, REPO_NAME, MODEL_NAME")
            raise EnvironmentError(
                "Error while loading env variables: DAGSHUB_TOKEN, REPO_OWNER, REPO_NAME, MODEL_NAME")

        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_TOKEN
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

        dagshub_url = "https://dagshub.com"
        # setting up mlflow tracking url
        mlflow.set_tracking_uri(f"{dagshub_url}/{REPO_OWNER}/{REPO_NAME}")

        # loading the model from mlflow registry
        cls.new_model_name = MODEL_NAME
        cls.new_model_version = cls.get_latest_model_version(
            cls.new_model_name)
        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # load the test_data
        cls.holdout_data = pd.read_csv("data/processed/test.csv")

    @staticmethod
    def get_lates_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        logger.info(
            f"latest_version: {latest_version} from get_lates_model_version() function is loaded.")
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        df = pd.read_csv("data/processed/sample_ml_dataset.csv")
        X = df.drop(columns=["view_count"])

        # predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(X)

        # verify the input shape
        self.assertEqual(X.shape[1], len(X))

        # verify the output shape
        self.assertEqual(len(prediction), X.shape[0])

    def test_model_performance(self):
        # extract features and labels from test data
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # calculating performance metrics for new model
        mean_absolute_error_new = mean_absolute_error(
            y_holdout, y_pred_new).round(2)
        mean_squared_error_new = mean_squared_error(
            y_holdout, y_pred_new).round(2)
        r2_score_new = r2_score(y_holdout, y_pred_new).round(2)

        # defining expected thresholds for performance metrics
        expected_mean_absolute_error = 0.78
        expected_mean_squared_error = 1.24
        expected_r2_score = 0.81

        # assert the new model meets the performance thresholds
        self.assertGreaterEqual(mean_absolute_error_new, expected_mean_absolute_error,
                                f"mean_absolute_error shold be atleast: {expected_mean_absolute_error}")

        self.assertGreaterEqual(mean_squared_error_new, expected_mean_squared_error,
                                f"mean_squared_error shold be atleast: {expected_mean_squared_error}")
        self.assertGreaterEqual(r2_score_new, expected_r2_score,
                                f"r2_score shold be atleast: {expected_r2_score}")


if __name__ == "__main__":
    unittest.main()
