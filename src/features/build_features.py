import pandas as pd
import numpy as np
import yaml
from src.logger.logger import configure_logger
from src.utils.yaml_loader import yaml_loader
from sklearn.model_selection import train_test_split

logger= configure_logger()

def train_test_data_split(file_path: str,test_size:float)->None:
    """ Function to split the data into train.csv and test.csv to avoid data leaking."""

    try: 
        df= pd.read_csv(file_path)
        if not df.empty:
            logger.info(f"Data is loaded successfully. Rows: {len(df)}")
        else:
            logger.exception(f"Data loading error. Rows: {len(df)}")
            raise  
        
        train_df, test_df = train_test_split(df,test_size=test_size,random_state=42)

        if not train_df.empty:
            logger.info(f"Data is loaded successfully. Rows of train_df: {len(train_df)}")
        else:
            logger.exception(f"Data loading error. Rows of train_df: {len(train_df)}")
            raise
        
        if not test_df.empty:
            logger.info(f"Data is loaded successfully. Rows of train_df: {len(test_df)}")
        else:
            logger.exception(f"Data loading error. Rows of train_df: {len(test_df)}")
            raise
        #saving the files
        train_df.to_csv("./data/processed/train.csv",index=None)    
        test_df.to_csv("./data/processed/test.csv",index=None)  

    except FileNotFoundError as e:
        logger.exception(f"File not found: {e}")  
        raise          
    except pd.errors.ParserError as e:
        logger.exception(f"Error while parsing the data: {e}")  
        raise  
    except Exception as e:
        logger.exception(f"Un excepted error has occurred: {e}")  
        raise  
    
def main():
    file_path= './data/interim/interim.csv'
    test_size= yaml_loader("./params.yaml")["data_ingestion"]["test_size"]
    train_test_data_split(file_path,test_size)
    
       
if __name__=="__main__":
    main()