import pandas as pd
import numpy as np
import yaml
from src.utils.yaml_loader import yaml_loader

from src.logger.logger import configure_logger

logger = configure_logger()


def data_validation():
    """ Function to test the required columns existing or not."""
    try:
        logger.info("Starting data ingestion and reading the file.")

        file_path = "./data/raw/youtube_10000_videos.csv"
        df = pd.read_csv(file_path)

        logger.info(f"File is loaded.Number of records: {len(df)}")

        required_columns = yaml_loader("./params.yaml")["required_columns"]
        logger.info(
            f"Required columns data is loaded: {len(required_columns)}")

        for column in required_columns:
            if not column in df.columns:
                logger.exception("Required columns are missing.")
                raise
        logger.info("Data validation is successful")

    except pd.errors.ParserError as error:
        logger.exception(f"Error while parsing the input file: {error}")
        raise
    except yaml.error.YAMLError as error:
        logger.exception(f"Error while reading the params.yaml file: {error}")
        raise
    except Exception as e:
        logger.exception(f"unexpected error: {e}")
        raise


if __name__ == "__main__":
    data_validation()
