# import pickle


# def save_model(model_save_path: str, model_pipeline) -> None:
#     """
#     Function to load the model.
#     parameters:
#     inputs:
#     model_save_path:str
#     outputs:
#     None
#     """
#     try:
#         with open(model_save_path, "wb") as model_dump:
#             pickle.dump(model_pipeline, model_dump)
#     except Exception as e:
#         raise e

import pickle
from pathlib import Path
from typing import Any


def save_model(model_save_path: str, model_pipeline: Any) -> None:
    """Serialize a fitted model, creating the destination directory if needed."""
    destination = Path(model_save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file first so a failed serialization does not leave a
    # partially written model artifact at the destination path.
    temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary_destination.open("wb") as model_dump:
            pickle.dump(model_pipeline, model_dump, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_destination.replace(destination)
    except Exception:
        temporary_destination.unlink(missing_ok=True)
        raise
