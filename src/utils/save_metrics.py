from src.logger.logger import configure_logger
import json

logger= configure_logger()

def save_metrics(metrics:dict,file_path:str):
    """ Function to save the metrics as json file using the given metrics as dictionary."""
    try:
        indent= len(metrics)

        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=indent)
        logger.info(f"metrics saved to path: {file_path}")    
        
    except Exception as e:
        logger.exception("Error occurred.")  
        raise      

if __name__=="__main__":
    metrics={}
    file_path=""   
    save_metrics(metrics,file_path)      