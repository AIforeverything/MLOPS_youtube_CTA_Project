import pickle
from pathlib import Path
from typing import Any


def save_model(model_save_path: str, model_pipeline: Any) -> None:
    try:
        model_path = Path(model_save_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as model:
            pickle.dump(model_pipeline, model, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise
