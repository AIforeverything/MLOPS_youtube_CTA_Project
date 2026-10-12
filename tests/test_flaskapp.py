"""Tests for the Flask YouTube view-count prediction endpoint."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from flask_app.app import app


# Resolve data paths relative to the repository, not the shell's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data" / "processed" / "sample_ml_dataset.csv").exists():
    # Supports the common layout where this file is stored in tests/.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sample_ml_dataset.csv"


class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<title>YouTube View Count Predictor</title>", response.data)

    def test_predict_page(self):
        self.assertTrue(
            SAMPLE_DATA_PATH.exists(),
            f"Sample dataset was not found at: {SAMPLE_DATA_PATH}",
        )
        df = pd.read_csv(SAMPLE_DATA_PATH)
        self.assertIn("view_count", df.columns, "Sample dataset must contain the target column 'view_count'.")

        # The endpoint predicts one item per request. Convert the first row to
        # JSON-safe Python values instead of passing a DataFrame to Flask.
        features = df.drop(columns=["view_count"])
        self.assertGreater(len(features), 0, "Sample dataset contains no rows.")
        sample = json.loads(features.iloc[0].to_json())

        response = self.client.post("/predict", json=sample)

        self.assertEqual(
            response.status_code,
            200,
            f"Prediction endpoint returned {response.status_code}: {response.get_data(as_text=True)}",
        )
        self.assertTrue(response.is_json, "Prediction endpoint should return JSON.")
        result = response.get_json()
        self.assertIsInstance(result, dict)
        self.assertIn("prediction", result)
        self.assertIsInstance(result["prediction"], (int, float))
        self.assertTrue(np.isfinite(result["prediction"]))


if __name__ == "__main__":
    unittest.main()
