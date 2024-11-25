'''
Centralized logging for this project

Logs to a file and the console for every run for redundancy.
'''

import logging
import os
from datetime import datetime 

def create_logger(logger_name):
    '''
    Creates the global logger object
    '''
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    log_name = f"logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fh = logging.FileHandler(log_name)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

LOGGER_NAME = "Project 2"
logger = create_logger(LOGGER_NAME)