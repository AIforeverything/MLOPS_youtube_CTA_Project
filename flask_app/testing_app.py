from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__name__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object containing prediction features"}), 400

    required_features = [
        "category",
        "subscriber_count",
        "channel_view_count",
        "duration_seconds",
    ]

    missing = [name for name in required_features if name not in payload]
    if missing:
        return jsonify({"error": f"Missing required feature(s):{','.join(missing)}"}), 400

    try:
        # one row one prediction request
        input_df = pd.DataFrame([{
            "category": str(payload["category"]),
            "subscriber_count": float(payload["subscriber_count"]),
            "channel_view_count": float(payload["channel_view_count"]),
            "duration_seconds": float(payload["duration_seconds"]),
        }])
        print(input_df)

        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": float(prediction)})

    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the Flask server logs."}), 500


if __name__ == "__main__":
    # Disable the debug reloader so it doesn't import the app/model a second time.
    app.run(debug=True, use_reloader=False)
