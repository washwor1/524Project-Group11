import logging
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import time
import random 
import sys

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from classical_models import rf, svm, lstm, gaussian

from modeling import TransformerModel, ClassicalModels
from feature_engineering import extract_features
from dataset_handling import book_train_test_split, load_dataset
from p_logging import logger

LOGGER_NAME = "proj2_logger"



def experiment(config_name):
    # load dataset
    # run train-test-split on df (will produce label column)
    df = load_dataset(config_name=config_name)
    logger.info("Successfully read in dataset, now performing train-test-split")
    if config_name == "primary_authors":
        df = book_train_test_split(df)
    elif config_name == "all":
        df = book_train_test_split(df, margin_of_error=0.01, initial_growth=5, growth=100)
    # models = [TransformerModel(), ClassicalModels()]
    models = [ClassicalModels()]
    metrics = []
    for model in models:
        model.create_features(df, config_name)
        model.fit()
        metrics += model.predict()

    metrics_df = pd.DataFrame(metrics, columns=['model_name', 'data_type', 'phase', 'duration', 'accuracy', 'f1-score', 'precision', 'recall'])
    metrics_df.to_csv(f"data/{config_name}/metrics.csv", index=False)
if __name__ == "__main__":
    CONFIG_NAME = "all"
    if len(sys.argv) == 2:
        CONFIG_NAME = str(sys.argv[1])
    
    logger.info(f"Running models (config = '{CONFIG_NAME}')")
    start = time.time()
    experiment(CONFIG_NAME)
    logger.info(f"Finished running experiment (took {(time.time() - start):.4f} seconds)")
