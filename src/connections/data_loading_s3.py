import os
from io import BytesIO

import boto3
import pandas as pd
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)
from dotenv import load_dotenv

from src.utils.logger import configure_logger


# Configure application logger
logger = configure_logger()


# Load environment variables from .env
load_dotenv()

# Read environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
FILE_KEY = os.getenv("FILE_KEY")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "us-east-1")


def validate_environment_variables():
    """
    Validate that all required environment variables are available.
    """

    required_variables = {
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "S3_BUCKET_NAME": S3_BUCKET_NAME,
        "FILE_KEY": FILE_KEY,
        "AWS_REGION_NAME": AWS_REGION_NAME,
    }

    missing_variables = [
        variable_name
        for variable_name, variable_value in required_variables.items()
        if not variable_value
    ]

    if missing_variables:
        error_message = (
            "The following environment variables are missing: "
            + ", ".join(missing_variables)
        )

        logger.error(error_message)
        raise ValueError(error_message)


class AWSS3Connections:
    """
    Class responsible for connecting to Amazon S3 and fetching CSV files.
    """

    def __init__(
        self,
        bucket_name,
        aws_access_key_id,
        aws_secret_access_key,
        aws_region,
    ):
        """
        Initialize the S3 connection.

        Parameters
        ----------
        bucket_name : str
            Name of the S3 bucket.

        aws_access_key_id : str
            AWS access key ID.

        aws_secret_access_key : str
            AWS secret access key.

        aws_region : str
            AWS region, for example: ap-south-1.
        """

        self.bucket_name = bucket_name
        self.aws_region = aws_region

        try:
            self.s3_client = boto3.client(
                service_name="s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region,
            )

            logger.info(
                f"S3 client initialized successfully for bucket '{self.bucket_name}' "
                f"in region {self.aws_region}."
                # self.bucket_name,
                # self.aws_region,
            )

        except Exception:
            logger.exception("Failed to initialize the S3 client.")
            raise

    def fetch_file_from_s3(self, file_key):
        """
        Fetch a CSV file from S3 and return it as a Pandas DataFrame.

        Parameters
        ----------
        file_key : str
            S3 object key, for example: data/train.csv.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the CSV data.
        """

        try:
            logger.info(
                f"Fetching file '{file_key}' from bucket '{self.bucket_name}'.",
                # file_key,
                # self.bucket_name,
            )

            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key,
            )

            # Read the S3 response body directly into Pandas.
            df = pd.read_csv(BytesIO(response["Body"].read()))

            logger.info(
                f"Successfully loaded '{file_key}'. Number of records: {len(df)}.",
                # file_key,
                # len(df),
            )

            return df

        except self.s3_client.exceptions.NoSuchKey:
            logger.exception(
                f"The S3 object '{file_key}' does not exist in bucket '{self.bucket_name}'.",
                # file_key,
                # self.bucket_name,
            )
            raise

        except ClientError as error:
            error_code = error.response.get(
                "Error",
                {},
            ).get(
                "Code",
                "Unknown",
            )

            logger.exception(
                f"AWS error while fetching '{file_key}'. Error code: {error_code}.",
                # file_key,
                # error_code,
            )
            raise

        except UnicodeDecodeError:
            logger.exception(
                f"The file '{file_key}' could not be decoded as UTF-8.",
                # file_key,
            )
            raise

        except pd.errors.ParserError:
            logger.exception(
                f"The file '{file_key}' is not a valid CSV file.",
                # file_key,
            )
            raise

        except Exception:
            logger.exception(
                f"Unexpected error while fetching '{file_key}' from S3.",
                # file_key,
            )
            raise


def main():
    """
    Main execution function.
    """

    validate_environment_variables()

    logger.info("Starting S3 data-ingestion process.")

    data_ingestion = AWSS3Connections(
        bucket_name=S3_BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_region=AWS_REGION_NAME,
    )

    dataframe = data_ingestion.fetch_file_from_s3(
        file_key=FILE_KEY,
    )

    print("\nData was fetched successfully.")
    print(f"Number of records: {len(dataframe)}")
    print("\nFirst five records:")
    print(dataframe.head())
    data_save_local='data/raw/youtube_10000_videos.csv'
    dataframe.to_csv(data_save_local,index=False)

    logger.info("S3 data-ingestion process completed successfully.")
    logger.info(f"""{FILE_KEY} is saved to local as "{data_save_local}" for experimentation.""")


if __name__ == "__main__":
    try:
        main()

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
            # error,
        )
        raise

    except Exception:
        logger.exception("S3 data-ingestion process failed.")
        raise