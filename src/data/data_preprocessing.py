import pandas as pd 
import numpy as np 
import yaml
from src.utils.yaml_loader import yaml_loader
from src.logger.logger import configure_logger
from src.utils.remove_null_values import remove_null

logger= configure_logger()

def data_processing(file_path:str,file_saving_path:str,required_columns:list[str]):
    """This function processes the data.It removes null values from the required columns"""
    try:
        logger.info("Starting data preprocessing and reading the file.")
        
        
        df= pd.read_csv(file_path)
        df= df[required_columns]
        df1= remove_null(df)
        
        logger.info(f"Null values removed: {len(df)-len(df1)}")
        
        df1.to_csv(file_saving_path,index=False)
        logger.info("Data preprocessing is successful")
       
    except pd.errors.ParserError as error:
        logger.exception(f"Error while parsing the input file: {error}")   
        raise 
    except yaml.error.YAMLError as error:
        logger.exception(f"Error while reading the params.yaml file: {error}")   
        raise 
    except Exception as e:
        logger.exception(f"unexpected error: {e}")
        raise
    
def main():
    file_path= "./data/raw/youtube_10000_videos.csv"
    file_saving_path= "./data/interim/interim.csv"
    required_columns= yaml_loader("./params.yaml")["required_columns"]
    data_processing(file_path,file_saving_path,required_columns)   
    
        
if __name__=="__main__":
    main()