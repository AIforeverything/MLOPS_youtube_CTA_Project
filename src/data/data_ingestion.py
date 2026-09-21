import pandas as pd
import numpy as np
from src.logger.logger import configure_logger
from src.connections.data_loading_s3 import main
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    PartialCredentialsError
)

logger= configure_logger()

def data_ingestion()->pd.DataFrame:
    """ Function to read the data file and returns pandas dataframe"""
    try:
        df= main()
        # print(df)
        return df
    except NoCredentialsError:
        logger.error(
            "AWS credentials were not found. "
            "Check your .env file or AWS credential configuration."
        )
        raise

    except PartialCredentialsError:
        logger.error(
            "Incomplete AWS credentials were provided. "
            "Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        )
        raise

    except ClientError as error:
        logger.error(
            f"AWS ClientError occurred: {error}",
        )
        raise

    except Exception:
        logger.exception("S3 data-ingestion process failed.")
        raise
    
if __name__=="__main__":
    data_ingestion()
            
    