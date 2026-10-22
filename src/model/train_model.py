# import pandas as pd
# import numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# import yaml
# from src.logger.logger import configure_logger
# from src.utils.combined_pipeline import combined_transform, model_with_combined_processor
# from src.utils.save_model import save_model
# from src.utils.yaml_loader import yaml_loader
# logger = configure_logger()


# def train_model(file_path: str,
#                 model_pipeline: Pipeline,
#                 save_model_path: str) -> None:
#     try:
#         df = pd.read_csv(file_path)
#         X = df.drop(columns=["view_count"])
#         y = np.log1p(df["view_count"])

#         model_pipeline.fit(X, y)
#         save_model(save_model_path, model_pipeline)

#         if model_pipeline:
#             logger.info("model saved successfully!")
#         else:
#             logger.info("model not saved successfully!")
#             raise

#     except FileNotFoundError as e:
#         logger.exception(f"File was not found: {e}")
#         raise

#     except pd.errors.ParserError as e:
#         logger.exception(f"File parsing error: {e}")
#         raise
#     except yaml.error.YAMLError as e:
#         logger.exception(f"yaml file parsing error: {e}")
#         raise
#     except Exception as e:
#         logger.exception(f"un expected error occurred: {e}")


# def main():
#     file_path = "./data/processed/train.csv"

#     numeric_features = yaml_loader("./params.yaml")["numeric_features"]
#     categorical_features = yaml_loader("./params.yaml")["categorical_features"]
#     combined_processor_pipeline = combined_transform(
#         numeric_features, categorical_features)

#     best_params = yaml_loader("./params.yaml")["best_params_RandomForest"]

#     model = RandomForestRegressor(
#         n_estimators=best_params["n_estimators"],
#         max_depth=best_params["max_depth"],
#         min_samples_split=best_params["min_samples_split"],
#         random_state=42, n_jobs=-1, verbose=1)

#     model_pipeline = model_with_combined_processor(
#         combined_processor_pipeline, model)


#     save_model_path = './models/model.pkl'

#     train_model(file_path, model_pipeline, save_model_path)
#     return train_model


# if __name__ == "__main__":
#     main()

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.logger.logger import configure_logger
from src.utils.combined_pipeline import (
    combined_transform,
    model_with_combined_processor,
)
from src.utils.save_model import save_model
from src.utils.yaml_loader import yaml_loader

logger = configure_logger()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_model(
    file_path: str,
    model_pipeline: Pipeline,
    save_model_path: str,
) -> None:
    """Fit the model pipeline and save the fitted model artifact."""
    try:
        data_path = Path(file_path)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path

        if not data_path.is_file():
            raise FileNotFoundError(f"Training data file does not exist: {data_path}")

        df = pd.read_csv(data_path)
        if "view_count" not in df.columns:
            raise ValueError(
                "Training dataset must contain the 'view_count' target column."
            )
        if df.empty:
            raise ValueError(f"Training dataset is empty: {data_path}")

        X = df.drop(columns=["view_count"])
        # The model is trained on log1p(target); prediction consumers must apply
        # expm1 to return predictions to the original view-count scale.
        y = np.log1p(df["view_count"])

        model_pipeline.fit(X, y)

        artifact_path = Path(save_model_path)
        if not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
        save_model(str(artifact_path), model_pipeline)
        logger.info("Model trained and saved successfully to %s", artifact_path)

    except (FileNotFoundError, ValueError, pd.errors.ParserError, yaml.error.YAMLError):
        logger.exception("Model training failed due to invalid input or configuration.")
        raise
    except Exception:
        logger.exception("Unexpected error occurred during model training.")
        raise


def main() -> None:
    params_path = PROJECT_ROOT / "params.yaml"
    params = yaml_loader(str(params_path))

    file_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    numeric_features = params["numeric_features"]
    categorical_features = params["categorical_features"]
    combined_processor_pipeline = combined_transform(
        numeric_features, categorical_features
    )

    best_params = params["best_params_RandomForest"]
    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    model_pipeline = model_with_combined_processor(combined_processor_pipeline, model)
    save_model_path = PROJECT_ROOT / "models" / "model.pkl"
    train_model(str(file_path), model_pipeline, str(save_model_path))


if __name__ == "__main__":
    main()
