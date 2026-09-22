import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import yaml
from src.logger.logger import configure_logger
from src.utils.combined_pipeline import combined_transform, model_with_combined_processor
from src.utils.save_model import save_model
from src.utils.yaml_loader import yaml_loader
logger = configure_logger()


def train_model(file_path: str,
                model_pipeline: Pipeline,
                save_model_path: str) -> None:
    try:
        df = pd.read_csv(file_path)
        X = df.drop(columns=["view_count"])
        y = np.log1p(df["view_count"])

        model_pipeline.fit(X, y)
        save_model(save_model_path, model_pipeline)

        if model_pipeline:
            logger.info("model saved successfully!")
        else:
            logger.info("model not saved successfully!")
            raise

    except FileNotFoundError as e:
        logger.exception(f"File was not found: {e}")
        raise

    except pd.errors.ParserError as e:
        logger.exception(f"File parsing error: {e}")
        raise
    except yaml.error.YAMLError as e:
        logger.exception(f"yaml file parsing error: {e}")
        raise
    except Exception as e:
        logger.exception(f"un expected error occurred: {e}")


def main():
    file_path = "./data/processed/train.csv"

    numeric_features = yaml_loader("./params.yaml")["numeric_features"]
    categorical_features = yaml_loader("./params.yaml")["categorical_features"]
    combined_processor_pipeline = combined_transform(
        numeric_features, categorical_features)

    best_params = yaml_loader("./params.yaml")["best_params_RandomForest"]

    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        random_state=42, n_jobs=-1, verbose=1)

    model_pipeline = model_with_combined_processor(
        combined_processor_pipeline, model)

    save_model_path = './models/model.pkl'

    train_model(file_path, model_pipeline, save_model_path)
    return train_model


if __name__ == "__main__":
    main()
