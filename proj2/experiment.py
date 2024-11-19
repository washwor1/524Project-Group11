import logging
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import random

from feature_engineering import extract_features
from dataset_handling import book_train_test_split, load_dataset

logger = logging.getLogger("proj2_logger")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('test.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
import time
start = time.time()
logger.info("Beginning experiment")
df = book_train_test_split(load_dataset())
logger.info(f"Finished loading dataset (took {(time.time() - start)}")
tfidf, vecs = extract_features(df)
