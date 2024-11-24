'''
Create Dataset Script

Contains functions to download raw data, process it, and combine it into a single parquet file

Usage: python create_dataset [config_name]
- Runs the dataset creation (assumes spgc_raw.zip has been downloaded) for that config name
'''


import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zipfile
import gdown
from datetime import datetime
import time
import os
import sys

from p_logging import logger
import dask_cudf
from dask.dataframe import from_pandas
from tqdm.auto import tqdm
from dataset_handling import get_config_metadata, write_config_metadata
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import CountVectorizer
from timeit import default_timer as timer
from pyspark.sql.functions import udf, explode, split
from pyspark.sql.types import DoubleType, ArrayType, StringType

# Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
# (can use `tqdm.gui.tqdm`, `tqdm.notebook.tqdm`, optional kwargs, etc.)
tqdm.pandas(desc="Dataset Progress")


def unzip_archive(file, output_path):
    with zipfile.ZipFile(file, "r") as zip_fp:
        zip_fp.extractall(output_path)

def download_gdrive_file(file_id, output_file):
    '''
    Downloads dataset stored on Google Drive
    '''
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)

def process_metadata(metadata_path="data/spgc/metadata/metadata.csv"):
    '''
    Apply filters to the metadata to downselect to the best number of books
    '''
    start = time.time()
    logger.info("Processing metadata...")
    df = pd.read_csv(metadata_path)
    # include English-only books (Leaves ~59633 and removes ~14910)
    df['is_english'] = df['language'].apply(lambda x: x == "['en']")
    # remove non-text entries and non-English books (leaves 58,477)
    df = df[df['is_english'] & (df['type'] == 'Text')]
    df['pg_code'] = df['id'].apply(lambda x: int(x[2:]))
    # remove books with author as "Various" or "Anonymous" or "Unknown" (removes 4353 + 118)
    df = df[~(df['author'] == 'Various') & ~(df['author'] == 'Anonymous') & ~(df['author'] == 'Unknown')]
    # as of now, I am going to keep books with more than one author and just treat the group as one author. We can fix this later 
    # if this confounds the model
    df = df.dropna(axis=0, subset=['author', 'title'])
    df = df.sort_values('pg_code').reset_index(drop=True)
    df['author'] = df['author'].astype('category')
    df['author_id'] = df['author'].cat.codes
    df['book_id'] = df.groupby('author_id').cumcount()
    # remove authors with only one book (19339 authors -> 5899 authors; 51720 books -> 38280 books)
    book_counts = df.groupby('author_id')['book_id'].count()
    df = df[df['author_id'].isin(book_counts[book_counts > 1].index)]
    meta_df = df.drop(columns=['id', 'authoryearofbirth', 'authoryearofdeath', 'language', 'downloads', 'subjects', 'type', 'is_english']).reset_index(drop=True)
    logger.info(f"Finished processing metadata (took {time.time() - start} seconds)")
    return meta_df

def process_dataset(meta_df, data_dir="data/spgc/data/tokens", output_dir="data"):
    '''
    Process all the selected works in SPGC into a single Parquet file
    
    This will take a while. 
    '''
    conf = SparkConf().setMaster("local[*]").setAppName("SparkTFIDF").set('spark.driver.memory', '50G').set('spark.driver.maxResultSize', '20G')
    
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    arr = [] # columns: author_id, book_id, tokens
    failed_arr = [] # array of book codes that could not be found
    CHUNK_SIZE = 500
    def read_book(pg_code):
        data_dir = "data/spgc/data/tokens"
        nonlocal failed_arr
        tokens = []
        try:
            with open(f"{data_dir}/PG{pg_code}_tokens.txt", 'r', encoding='utf-8') as fp:
                tokens = fp.read().splitlines()
        except FileNotFoundError:
            failed_arr.append(pg_code)
    
        return [' '.join(tokens[i:(i + CHUNK_SIZE)]) for i in range(0, len(tokens), CHUNK_SIZE)]
    df = spark.createDataFrame(meta_df)
    udf_read_book = udf(read_book, ArrayType(StringType()))
    df = df.withColumn("text", explode(udf_read_book(df.pg_code)))
    logger.info("Finished processing dataset")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Saving dataset...")
    df.select('author_id', 'book_id', 'text').write.parquet( f'{output_dir}/dataset.parquet', mode='overwrite', partitionBy='author_id')
    
    data = get_config_metadata(output_dir)
    data[0] = int(time.mktime(datetime.now().timetuple()))
    write_config_metadata(data, output_dir)
    # remove missing records from metadata so only included books are recorded
    meta_df = meta_df[~meta_df['pg_code'].isin(failed_arr)].reset_index(drop=True)
    meta_df.to_csv(f"{output_dir}/metadata.csv", index=False)
    logger.info("Saved dataset, updated metadata")
    spark.stop()


if __name__ == "__main__":
    # code to download dataset
    # will download an ~8 GB file, so it will take second
    # download_gdrive_file("1VJcL_0B-7YcAkaSTXnHOKXLa_EAbmpCK", "data/spgc_raw.zip")
    # unzip_archive("data/spgc_raw.zip", "data/spgc/")
    
    # CONFIG_NAME = "primary_authors"
    CONFIG_NAME = "all"
    if len(sys.argv) == 2:
        CONFIG_NAME = str(sys.argv[1])
    logger.info(f"Creating dataset (config = '{CONFIG_NAME}')")
    meta_df = process_metadata()
    # combine all three files into a single parquet file
    if CONFIG_NAME == "primary_authors":
        meta_df = meta_df[meta_df['author'].isin(['Leblanc, Maurice', 'Christie, Agatha', 'Chesterton, G. K. (Gilbert Keith)', 'Doyle, Arthur Conan'])]
    
    process_dataset(meta_df, output_dir=f"data/{CONFIG_NAME}")
