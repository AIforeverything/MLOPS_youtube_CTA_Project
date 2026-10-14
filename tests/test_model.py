"""Model-registry, input-signature, and holdout-performance tests."""

import os
import unittest
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger.logger import configure_logger

logger = configure_logger()

# Resolve data paths relative to the repository, independent of the current directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data" / "processed" / "test.csv").exists():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        token = os.getenv("DAGSHUB_TOKEN")
        repo_owner = os.getenv("REPO_OWNER")
        repo_name = os.getenv("REPO_NAME")
        model_name = os.getenv("MODEL_NAME")

        missing = [
            name
            for name, value in {
                "DAGSHUB_TOKEN": token,
                "REPO_OWNER/repo_owner": repo_owner,
                "REPO_NAME/repo_name": repo_name,
                "MODEL_NAME/model_name": model_name,
            }.items()
            if not value
        ]
        if missing:
            raise EnvironmentError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Set them in the repository .env file or the shell environment."
            )

        # DagsHub MLflow authentication: repository owner as username, token as password.
        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)

        cls.new_model_name = model_name
        cls.new_model_version = cls.get_latest_model_version(
            cls.new_model_name)
        if cls.new_model_version is None:
            raise RuntimeError(
                f"No model versions found for registered model '{cls.new_model_name}' in DagsHub. "
                "Check MODEL_NAME and confirm that the model is registered."
            )

        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        logger.info("Loading registered model from %s", cls.new_model_uri)
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError(
                f"Holdout test dataset not found: {test_path}")
        cls.holdout_data = pd.read_csv(test_path).dropna()
        if cls.holdout_data.empty:
            raise ValueError(f"Holdout test dataset is empty: {test_path}")

    # @staticmethod
    # def get_latest_model_version(model_name):
    #     """Prefer Staging, then Production, then an un-staged/latest version."""
    #     client = mlflow.MlflowClient()
    #     for stage in ("Staging", "Production", "None"):
    #         versions = client.get_latest_versions(model_name, stages=[stage])
    #         if versions:
    #             version = max(versions, key=lambda item: int(item.version)).version
    #             logger.info("Selected model version %s from stage %s", version, stage)
    #             return version

    #     # Last-resort fallback for registry versions whose stage is not returned
    #     # by the stage queries above.
    #     versions = client.search_model_versions(f"name='{model_name}'")
    #     if versions:
    #         return max(versions, key=lambda item: int(item.version)).version
    #     return None

    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()

        versions = client.search_model_versions(
            f"name = '{model_name}'"
        )

        if not versions:
            return None

        latest_version = max(
            versions,
            key=lambda version: int(version.version)
        )

        logger.info(
            "Latest registered version for %s: %s",
            model_name,
            latest_version.version,
        )

        return latest_version.version

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def _predict_with_compatibility_diagnostic(self, features, dataset_label):
        """Predict and explain common sklearn pickle-version incompatibilities."""
        try:
            return np.asarray(self.new_model.predict(features)).reshape(-1)
        except AttributeError as exc:
            if "_fill_dtype" in str(exc) or "_fit_dtype" in str(exc):
                import sklearn

                raise AssertionError(
                    "The registered model cannot run with the installed scikit-learn "
                    f"version ({sklearn.__version__}). The test log reports that the "
                    "serialized SimpleImputer was trained with scikit-learn 1.7.1. "
                    "Install scikit-learn==1.7.1 in this environment, or retrain and "
                    "re-register the model using the same scikit-learn version used "
                    f"for serving. Failure occurred on the {dataset_label}."
                ) from exc
            raise

    def test_model_signature(self):
        sample_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
        self.assertTrue(
            sample_path.exists(), f"Sample dataset not found: {sample_path}"
        )
        df = pd.read_csv(sample_path)
        self.assertIn(
            "view_count", df.columns, "Sample dataset must contain 'view_count'."
        )
        X = df.drop(columns=["view_count"])

        prediction = self._predict_with_compatibility_diagnostic(
            X, "sample dataset")
        self.assertEqual(
            prediction.shape[0],
            X.shape[0],
            "Prediction count must match input row count.",
        )
        self.assertTrue(
            np.isfinite(prediction.astype(float)).all(),
            "Predictions should be finite numbers.",
        )

        signature = self.new_model.metadata.signature
        if signature is not None and signature.inputs is not None:
            expected_input_count = len(signature.inputs)
            self.assertEqual(
                X.shape[1],
                expected_input_count,
                f"Model signature expects {expected_input_count} input columns, but dataset has {X.shape[1]}. "
                "Check that the sample data columns match the training features.",
            )

    def test_model_performance(self):
        if "view_count" not in self.holdout_data.columns:
            self.fail(
                "Holdout dataset must contain the target column 'view_count'.")

        X_holdout = self.holdout_data.drop(columns=["view_count"])
        y_holdout = self.holdout_data["view_count"]
        y_holdout = np.log1p(y_holdout)
        y_pred = self._predict_with_compatibility_diagnostic(
            X_holdout, "holdout dataset"
        )

        self.assertEqual(
            len(y_pred),
            len(y_holdout),
            "Prediction count must match holdout target count.",
        )
        self.assertTrue(
            np.isfinite(y_pred.astype(float)).all(),
            "Model predictions contain NaN or infinity.",
        )

        mae = mean_absolute_error(y_holdout, y_pred)
        mse = mean_squared_error(y_holdout, y_pred)
        r2 = r2_score(y_holdout, y_pred)
        logger.info(
            "Holdout metrics: MAE=%.4f, MSE=%.4f, R2=%.4f", mae, mse, r2)

        # Optional project-specific quality gates. MAE and MSE are errors, so
        # they must be <= their maximum thresholds; R2 must be >= its minimum.
        max_mae = os.getenv("MAX_MAE")
        max_mse = os.getenv("MAX_MSE")
        min_r2 = os.getenv("MIN_R2", "0.8")
        self.assertTrue(
            np.isfinite([mae, mse, r2]).all(
            ), "Evaluation metrics must be finite."
        )
        if max_mae is not None:
            self.assertLessEqual(
                mae, float(
                    max_mae), f"MAE {mae:.4f} exceeds MAX_MAE={max_mae}."
            )
        if max_mse is not None:
            self.assertLessEqual(
                mse, float(
                    max_mse), f"MSE {mse:.4f} exceeds MAX_MSE={max_mse}."
            )
        self.assertGreaterEqual(
            r2, float(min_r2), f"R2 {r2:.4f} is below MIN_R2={min_r2}."
        )


if __name__ == "__main__":
    unittest.main()
