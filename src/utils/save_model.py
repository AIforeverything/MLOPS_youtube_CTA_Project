import pickle
from pathlib import Path
from typing import Any


def save_model(model_save_path: str, model_pipeline: Any) -> None:
    try:
        with open(model_save_path, "wb") as model:
            pickle.dump(model_pipeline, model, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise
