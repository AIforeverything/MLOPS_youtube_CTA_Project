import unittest
import pandas as pd
import numpy as np
from flask_app.app import app


class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b'<title>YouTube View Count Predictor</title>', response.data)

    def test_predict_page(self):
        # Load sample dataset and remove target column
        response = self.client.post(
            '/predict',
            json=pd.read_csv("data/processed/sample_ml_dataset.csv").drop(columns=["view_count"]))

        # df = pd.read_csv(
        #     "data/processed/sample_ml_dataset.csv"
        # ).drop(columns=["view_count"])

        # # Use one sample for a single prediction request
        # sample = df.iloc[0].to_dict()

        # # Send features as JSON
        # response = self.client.post(
        #     "/predict",
        #     json=sample
        # )

        # Verify response status
        self.assertEqual(response.status_code, 200)

        # Verify JSON response
        self.assertEqual(response.content_type, "application/json")

        result = response.get_json()

        # Verify prediction exists
        self.assertIn("prediction", result)

        # Verify prediction is numeric and finite
        self.assertIsInstance(result["prediction"], (int, float))
        self.assertTrue(np.isfinite(result["prediction"]))


if __name__ == "__main__":
    unittest.main()
