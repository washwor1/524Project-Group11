# Project 2

## Downloading Data Files
All the datafiles for this project can be found here: [https://drive.google.com/drive/folders/1ticQ0gyPLCTGpLQgpI-jHCCL5dgOeIHO?usp=sharing](https://drive.google.com/drive/folders/1ticQ0gyPLCTGpLQgpI-jHCCL5dgOeIHO?usp=sharing). Many of the script pre-suppose that some files are already downloaded, so please review them before running the scripts. I've tried to denote all of them in the Usage section below.

## Dataset Configuration
This project utilizes (for now) two dataset configurations: `all` or `primary_authors`. The first should be pretty straightforward to understand; the second refers to only the books written by the four main authors used in Project 1. 

## Usage
- `python create_dataset.py [CONFIG_NAME]` - Create the dataset for a given config as the `data/[CONFIG_NAME]/dataset.parquet`.
- `python experiment.py [CONFIG_NAME]` - Runs a full experiment of Transformers + Classical Models on the given config. 
- `python feature_engineering.py [CONFIG_NAME]` - Extracts TF-IDF and Word Embedding features for a config

## Directory Breakdown
- `data/`
    - `[CONFIG_NAME]/` - Contains dataset and results for a specific dataset config.
        - `run_dates.txt` - Metadata to save time when creating features. Contains two numbers: the time the dataset was last updated, and the time when the features were last updated
        - `dataset.parquet` - Main datafile stored in compress Parquet format. 
        - `tfidf/` and `glove/` - Stores results from the tests for both feature types for the classical models
    - `spgc/` - Directory storing the raw corpus
- `logs/` - Storage location of the log files created on each run
- `dataset_handling.py` - Contains helper functions for reading the dataset(s) in and the logic for the custom train-test-split
- `proj_logging.py` - Centralized logging for this project (creates the files stored in `logs/`)
- `modeling.py` - Contains helper classes for running the models