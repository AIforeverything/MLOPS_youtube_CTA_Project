import logging 
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys
from pathlib import Path

# constants
LOG_DIR= 'logs'
MAX_LOG_SIZE= 1024*1024 # 1MB
BACKUP_COUNT= 2

# Create a timestamped log file
timestamp= datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

# log formatter
formatter= logging.Formatter("[%(asctime)s]"
                             "%(name)s - %(levelname)s -"
                             "%(filename)s:%(lineno)s -"
                             "%(message)s")


def configure_logger():
    """ 
    Configure application wise logging.
    
    Returns: 
        logging.Logger : The configured root logger
    """
    # making an object of the logger
    logger= logging.getLogger() 
    
    # Preventing duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    # it sets the level for logger
    logger.setLevel(logging.DEBUG)

    # project root relative to this file
    project_root= Path(__file__).resolve().parents[2] # above two directories, we ger the root directory
    log_dir= project_root/LOG_DIR
    log_dir.mkdir(parents=True,exist_ok=True)
    
    # time stamp for file
    log_file_path= log_dir/ f"{timestamp}.log"
    
    # making Rotating file handler and adding to logger
    file_handler= RotatingFileHandler(
        filename=log_file_path,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # making console / streaming handler and adding to logger
    console_handler= logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate=False 
    # propagate = False prevents a logger's messages from being processed by its ancestor loggers. 
    # Use it when we have configured our own handlers and want to avoid duplicate output.
    
    return logger
    
if __name__=="__main__":
    configure_logger()    