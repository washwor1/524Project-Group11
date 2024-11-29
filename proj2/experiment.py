'''
Central script to run the experiment for a dataset configuration.


'''
import pandas as pd
import time
import sys

from modeling import TransformerModel, ClassicalModels
from dataset_handling import book_train_test_split, load_dataset
from proj_logging import logger

def experiment(config_name):
    # load dataset
    df = load_dataset(config_name=config_name)
    logger.info("Successfully read in dataset, now performing train-test-split")
    # run train-test-split on df (will produce is_train column)
    if config_name == "primary_authors":
        df = book_train_test_split(df)
    elif config_name == "all":
        # this dataset is bigger so it needs different parameters
        df = book_train_test_split(df, margin_of_error=0.01, initial_growth=10, growth=100)
    
    # models = [TransformerModel()]
    models = [TransformerModel(), ClassicalModels()]
    metrics = []
    for model in models:
        model.create_features(df, config_name)
        model.fit()
        metrics += model.predict()
    
    # write out metrics to csv
    metrics_df = pd.DataFrame(metrics, columns=['model_name', 'data_type', 'phase', 'duration', 'accuracy', 'f1-score', 'precision', 'recall'])
    metrics_df.to_csv(f"data/{config_name}/metrics.csv", index=False)

if __name__ == "__main__":
    CONFIG_NAME = "all"
    if len(sys.argv) == 2:
        CONFIG_NAME = str(sys.argv[1])
    
    logger.info(f"Running models (config = '{CONFIG_NAME}')")
    start = time.time()
    try:
        experiment(CONFIG_NAME)
    except Exception as e:
        logger.error(e)
        raise e
    logger.info(f"Finished running experiment (took {(time.time() - start):.4f} seconds)")
