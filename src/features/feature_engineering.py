import pandas as pd
import numpy as np
import yaml
from src.logger.logger import configure_logger
from src.utils.yaml_loader import yaml_loader
from sklearn.model_selection import train_test_split

logger= configure_logger()

def train_test_data_split(file_path: str)->pd.DataFrame:
    """ Function to split the data into train.csv and test.csv to avoid data leaking."""

    try: 
        df= pd.read_csv('./data/interim.csv')
        if not df.empty():
            logger.info(f"Data is loaded successfully. Rows: {len(df)}")
        else:
            logger.exception(f"Data loading error. Rows: {len(df)}")
            raise  
        test_size_from_yaml= yaml_loader("./params.yaml")["data_ingestion"]["test_size"]
        train_df, test_df = train_test_split(df,test_size=test_size_from_yaml,random_state=42)

        if not train_df.empty():
            logger.info(f"Data is loaded successfully. Rows: {len(df)}")
        else:
            logger.exception(f"Data loading error. Rows: {len(df)}")
            raise
        #saving the files
        train_df.to_csv("./data/processed/train.csv")    
        test_df.to_csv("./data/processed/test.csv")  



    except pd.errors.ParserError as e:
        logger.exception(f"Error while parsing the data: {e}")  
        raise  
    except Exception as e:
        logger.exception(f"Unexcepted error has occurred: {e}")  
        raise  