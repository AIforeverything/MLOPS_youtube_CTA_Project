import pandas as pd 
import numpy as np
from src.utils.load_model import load_model
from src.logger.logger import configure_logger
import dagshub
import mlflow
import mlflow.sklearn
import os
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

logger= configure_logger()

def model_predict_mlflow(test_data_path:str, model_pipeline_path:str):
    try:
        logger.info("Test Data Loading is started.") 
        test_data_df= pd.read_csv(test_data_path)  
        if not test_data_df.empty: 
            logger.info("Test Data is loaded successfully!")
            
        X_test=test_data_df.drop(columns=["view_count"])
        y_test= np.log1p(test_data_df["view_count"])
        
        model= load_model(model_pipeline_path)
        if model:
            logger.info("Model pipeline is loaded successfully!")
        
        y_pred= model.predict(X_test)   
        
        mae= mean_absolute_error(y_test,y_pred)
        mse= mean_squared_error(y_test,y_pred)
        r2= r2_score(y_test,y_pred)
        
        
         
    except FileNotFoundError as e:
        logger.exception(f"File doesn't exist error")   
        raise 
        
    except pd.errors.ParserError as e:
        logger.exception("Data loading error")
        raise
    
    
    
def main():
    test_data_path= "./data/processed/test.csv"
    model_pipeline_path= "./models/model.pkl"
    model_predict_mlflow(test_data_path,model_pipeline_path)
    
    
if __name__=="__main__":
    main()  
