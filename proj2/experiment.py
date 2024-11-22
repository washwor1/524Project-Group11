import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import random
import time
import sys

from feature_engineering import extract_features
from dataset_handling import book_train_test_split, load_dataset
from p_logging import logger




if __name__ == "__main__":
    CONFIG_NAME = "all"
    if len(sys.argv) == 2:
        CONFIG_NAME = str(sys.argv[1])
    
    logger.info(f"Creating features (config = '{CONFIG_NAME}')")
    # start = time.time()
    # df = load_dataset(config_name=CONFIG_NAME)
    # logger.info(f"Finished loading dataset (took {(time.time() - start)} seconds)")
    start = time.time()
    # df = book_train_test_split(df)
    # skip this if stuff is already up to date
    tfidf, vecs = extract_features(f"data/{CONFIG_NAME}/dataset.parquet", config_name=CONFIG_NAME)
    logger.info(f"Finished extracting features (took {(time.time() - start)} seconds)")