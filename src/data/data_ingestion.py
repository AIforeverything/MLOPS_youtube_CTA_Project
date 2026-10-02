import pandas as pd
from src.logger.logger import configure_logger
from pathlib import Path

logger = configure_logger()


def data_ingestion(file_path: str) -> pd.DataFrame:
    """Function to read the data file and returns pandas dataframe"""
    try:
        file = Path(file_path)
        if file.exists():
            df = pd.read_csv(file)
        else:
            logger.exception("File wa not found.")
            raise FileNotFoundError
        if not df.empty:
            logger.info(f"File {file} is loaded.")
        return df
    except pd.errors.ParserError:
        logger.exception(
            f"error occurred while loading file from the path: {file_path}"
        )
        raise
    except Exception as e:
        logger.exception("Error occurred: {e}")
        raise e


if __name__ == "__main__":
    file_path = "data/raw/youtube_10000_videos.csv"
    data_ingestion(file_path)
