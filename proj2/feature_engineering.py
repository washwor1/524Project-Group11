'''
    Contains functions and code to generate TF-IDF and GloVe word embeddings
    (adapted from Project 1 for this course)
'''
import os
import sys
import requests
import zipfile
import time
from datetime import datetime
import warnings

import pandas as pd
from timeit import default_timer as timer

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import VectorAssembler, HashingTF, IDF, MinMaxScaler
from pyspark.sql.functions import udf, split, monotonically_increasing_id, explode, col, mean, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import BooleanType

from dataset_handling import load_dataset, book_train_test_split, get_config_metadata, write_config_metadata
from proj_logging import logger

warnings.simplefilter(action='ignore', category=FutureWarning)

multiclass=False
remove_out_of_vocab=False

def load_glove_embeddings(glove_file_path):
    '''
    Loads the glove embeddings from the .txt file into memory. Takes a while and needs several gb memory
    '''
    embeddings_index = {}

    # Lead the txt into mem
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split(' ')
            word = values[0]
            try:
                embedding = [float(x) for x in values[1:]]
                embeddings_index[word] = embedding
            except ValueError:
                continue
    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index

def ensure_glove_embeddings(glove_dir='glove', glove_file='glove.840B.300d.txt'):
    """
    Checks if the GloVe embeddings exist. If not, downloads and extracts them.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(glove_dir):
        os.makedirs(glove_dir)

    glove_path = os.path.join(glove_dir, glove_file)

    # Check if the GloVe file exists
    if not os.path.isfile(glove_path):
        logger.info("GloVe embeddings not found. Downloading...")

        # URL of the GloVe zip file
        url = 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip'
        zip_path = os.path.join(glove_dir, 'glove.840B.300d.zip')

        # Download the zip file with progress bar
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_length = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        done = int(50 * downloaded / total_length)
                        print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded / (1024 * 1024):.2f}/{total_length / (1024 * 1024):.2f} MB", end='')
        logger.info("\nDownload complete. Extracting...")

        # Extract the .txt file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(glove_dir)
        logger.info("Extraction complete.")

        # Delete the zip file
        os.remove(zip_path)
        logger.info("Zip file deleted.")
    else:
        logger.info("GloVe embeddings already exist.")

    
class FeatureAnalysis():
    '''
    Class used to manage the feature engineering data
    '''
    def __init__(self, df_path, data_dir="data", label_arr=None):
        self.data_dir = data_dir
        self.data_set_path = df_path
        self.ngram_range = (1, 2) #we are using unigram and bigram
        self.max_features = 100  #number of features we want from teh dataset as inputs for the model
        self.run_metadata = get_config_metadata(data_dir)
        self.labels_arr = label_arr
        self.are_features_updated = self.run_metadata[1] > self.run_metadata[0]
        self.spark = None

    def start_spark(self):
        '''
        Configure PySpark session and read in main dataset
        '''
        self.conf = SparkConf().setMaster("local[*]").setAppName("SparkTFIDF") \
            .set('spark.local.dir', '/media/volume/team11data/tmp') \
            .set('spark.driver.memory', '50G') \
            .set('spark.driver.maxResultSize', '25G') \
            .set('spark.executor.memory', '10G')
    
        self.sc = SparkContext(conf=self.conf)
        self.sc.setLogLevel("ERROR")
        self.spark = SparkSession(self.sc)

        self.data_set_rdd = self.spark.read.parquet(self.data_set_path).sort(['author_id', 'book_id'])
        self.data_set_rdd = self.data_set_rdd.withColumn("words", split(self.data_set_rdd.text, " "))
    
    def stop_spark(self):
        ''' Stops the Spark session '''
        if self.spark is not None:
            self.spark.stop()    

    def extract_ngram_tfidf_features(self):
        '''
        Extract TF-IDF Featuures for the given configuration
        '''
        if self.are_features_updated:
            # don't rerun this, just load up dataset
            logger.info("Features are up to date, skipping TF-IDF feature creation")
            return pd.read_parquet(f'{self.data_dir}/tfidf_features.parquet')
        if self.spark is None:
            self.start_spark()
        logger.info("Extracting TF-IDF features...")
        toptimer = timer()
        starttime = timer()
        res=None
        try:
            logger.debug(f"Time to startup spark {timer() - starttime}")
            
            starttime = timer()
            hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
            featurizedData = hashingTF.transform(self.data_set_rdd)
            logger.debug(f"Time to HashingTF {timer() - starttime}")

            starttime = timer()
            idf = IDF(inputCol="rawFeatures", outputCol="features")
            idfModel = idf.fit(featurizedData)
            rescaledData = idfModel.transform(featurizedData)
            logger.debug(f"Time to IDF {timer() - starttime}")

            # convert Spark Vector to array to simplify reading the dataset back in
            tfidf_features_df = rescaledData.select("author_id", vector_to_array("features", 'float32').alias("features"))

            if self.labels_arr is not None:
                # add is_train column to RDD
                def is_label_udf(arr):
                    def f(x, arr):
                        res = None
                        try:
                            res = arr[x-1]
                        except Exception as e:
                            logger.info(x)
                            logger.error(e)
                            raise e
                        return res
                    return udf(lambda x: f(x, arr), BooleanType())
                tfidf_features_df = tfidf_features_df.withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())))
                tfidf_features_df = tfidf_features_df.withColumn(
                    'is_train', 
                    is_label_udf(self.labels_arr)(col('id'))
                ).select("author_id", "features", "is_train")
            
            tfidf_features_df.write.parquet(f'{self.data_dir}/tfidf_features.parquet', mode="overwrite")
            res = tfidf_features_df.toPandas()
            logger.info(f"Time total {timer() - toptimer}")
        except Exception as e:
            logger.error(e)
            self.stop_spark()
            raise e
        if res is None:
            raise Exception("Output was not created correctly")
        return res

    
    
    def generate_glove_vecs(self, embeddings_index=None):
        '''
        Generates the glove vectors for each chapter in the dataset. 

        Saves them to a numpy array file 'document_embeddings.npy'
        '''
        glove_file_path = 'glove.840B.300d.txt'
        out_file = f'{self.data_dir}/document_embeddings.parquet'
        if self.are_features_updated:
            # skip loading 
            logger.info("Features are up to date, skipping GloVe generation")
            return pd.read_parquet(out_file)

        if self.spark is None:
            self.start_spark()
        # WILL DOWNLOAD 2GB FILE
        embed_df = None
        if not os.path.exists("./glove_embeddings.parquet"):
            logger.info("Creating Parquet version of GloVe Embeddings")
            ensure_glove_embeddings(glove_dir='./', glove_file=glove_file_path)
            embeddings_index = load_glove_embeddings(glove_file_path)
            embed_list = [Row(word=k, embedding=v) for k, v in embeddings_index.items()]
           
            embed_df = self.spark.createDataFrame(embed_list) 
            embed_df.write.parquet("./glove_embeddings.parquet")
            logger.info("Saved Parquet version of GloVe Embeddings to disk")
        else:
            # loading a Parquet version of this saves a lot of time
            embed_df = self.spark.read.parquet("./glove_embeddings.parquet")
            logger.info("Loaded embedding dataset")
        
        df = self.data_set_rdd.select('author_id', 'book_id', 'words').withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())))
        word_df = df.withColumn('word', explode(df.words))
        joined_df = word_df.select('id', 'word').join(embed_df, embed_df.word == word_df.word, "inner")
        
        split_to_cols = [col('embedding')[i].alias(f'v{i}') for i in range(0, 300)]
        d = joined_df.select([col('id')] + split_to_cols)
        
        avg_expr = [mean(col(f'v{i}')).alias(f'v{i}') for i in range(0, 300)]
        agg_df = d.groupby('id').agg(*avg_expr)
        assembler = VectorAssembler(
            inputCols=[f'v{i}' for i in range(0, 300)],
            outputCol='vecs',
            handleInvalid = "keep" # or skip
        )
        e = assembler.transform(agg_df).select(col("id").alias("e_id"), "vecs")
        
        scaler = MinMaxScaler(inputCol="vecs", outputCol="features")
        scalerModel = scaler.fit(e)
        scaled_e = scalerModel.transform(e).select(col('e_id'), vector_to_array("features", 'float32').alias("features"))

        final_df = df.join(scaled_e, df.id == scaled_e.e_id, "inner")
            
        if self.labels_arr is not None:
            def is_label_udf(arr):
                def f(x, arr):
                    res = None
                    try:
                        res = arr[x-1]
                    except Exception as e:
                        logger.info(x)
                        logger.error(e)
                        raise e
                    return res
                return udf(lambda x: f(x, arr), BooleanType())
            final_df = final_df.withColumn(
                'is_train', 
                is_label_udf(self.labels_arr)(col('id'))
                # udf(lambda x: b_var[x-1], BooleanType())('id')
            ).select("author_id", "features", "is_train")
        
        final_df.write.parquet(out_file, mode="overwrite", partitionBy="author_id")
        logger.debug(f"Wrote file to {out_file}")
        return final_df.toPandas()


def extract_features(data_path, config_name="all", data_dir="data", embeddings_index=None, label_col=None):
    global multiclass
    multiclass=False
    
    global remove_out_of_vocab
    remove_out_of_vocab=False
    config_dir = f"{data_dir}/{config_name}"
    fean = FeatureAnalysis(data_path, config_dir, label_arr=label_col)
    start_time = time.time()
    # IF YOU DONT HAVE THE GLOVE EMBEDDINGS, WILL DOWNLOAD 2GB FILE.
    embeddings_index = fean.generate_glove_vecs()
    # vecs = fean.generate_glove_vecs_with_tfidf(embeddings_index)
    logger.info(f"Finished getting word embeddings (took {time.time() - start_time} seconds)")
    start_time = time.time()
    tfidf_features = fean.extract_ngram_tfidf_features()
    fean.stop_spark()
    end_time = time.time()
    logger.info(f"Finished extracting features (took {(end_time - start_time)} seconds)")
    fean.run_metadata[1] = int(time.mktime(datetime.now().timetuple()))
    write_config_metadata(fean.run_metadata, config_dir)
    # Return embeddings index so they can be used in the UI
    return tfidf_features, embeddings_index

if __name__ == '__main__':
    CONFIG_NAME = "all"
    if len(sys.argv) == 2:
        CONFIG_NAME = str(sys.argv[1])
    
    logger.info(f"Creating features (config = '{CONFIG_NAME}')")
    # start = time.time()
    df = load_dataset(config_name=CONFIG_NAME)
    if CONFIG_NAME == "primary_authors":
        df = book_train_test_split(df)
    elif CONFIG_NAME == "all":
        df = book_train_test_split(df, margin_of_error=0.01, initial_growth=5, growth=100)    
        # logger.info(f"Finished loading dataset (took {(time.time() - start)} seconds)")
    start = time.time()
    # skip this if stuff is already up to date
    tfidf, vecs = extract_features(f"data/{CONFIG_NAME}/dataset.parquet", config_name=CONFIG_NAME, label_col=list(df.is_train))
    logger.info(f"Finished extracting features (took {(time.time() - start)} seconds)")
