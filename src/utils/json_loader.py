import json
from src.logger.logger import configure_logger

logger = configure_logger()


def load_json(file_path: str) -> dict:
    """ This function loads a json file and returns python dictionary."""
    try:
        with open(file=file_path, mode='r') as file:
            output = json.load(file)
        if output:
            logger.info(
                f"json file was loaded successfully! File Path: {file_path}")
        return output
    except FileNotFoundError:
        logger.exception("File not found.")
        raise
    except Exception as e:
        logger.exception("Unexcepted error has occurred")
        raise


if __name__ == "__main__":
    file_path = "./reports/experiment_info.json"
    print(load_json(file_path))
