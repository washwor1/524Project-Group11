import logging
import os
from datetime import datetime 

def create_logger(logger_name):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    log_name = f"logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_name)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

LOGGER_NAME = "Project 2"
logger = create_logger(LOGGER_NAME)