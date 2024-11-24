'''
    Contains functions and code to generate TF-IDF and GloVe word embeddings
    (adapted from Project 1 for this course)
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import glob
import os
import sys
from collections import OrderedDict
import requests
import zipfile
import time
from dataset_handling import load_dataset, book_train_test_split
from datetime import datetime
from p_logging import logger
from dataset_handling import get_config_metadata, write_config_metadata
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark import SparkConf
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import VectorAssembler, HashingTF, IDF, Tokenizer, MinMaxScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import CountVectorizer
from timeit import default_timer as timer
from pyspark.ml.stat import Summarizer
from pyspark.sql.functions import udf, split, monotonically_increasing_id, explode, col, mean, concat, lit, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, ArrayType, BooleanType, StringType, MapType, StructType, StructField
import warnings
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

def average_embeddings(embeddings):
    '''
    Simple function to average glove embeddings for a sequence
    '''
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        #Returns zeros if the embeddings are empty for some reason. 
        return np.zeros(300)  

# def get_document_embedding(words, embeddings_index, averaging_function=average_embeddings):
#     # Get the embedding for easch word
#     embeddings = []
#     count = 0
#     broke_words = []

#     # iterate through the words and get the embedding for each one then apply the averaging function to all
#     for word in words:
#         embedding = embeddings_index.get(word)
#         if embedding is not None:
#             embeddings.append(embedding)
#         else:
#             broke_words.append(word)
#             count += 1
#     return averaging_function(embeddings).tolist()
# def get_document_embedding(words, embeddings_index, averaging_function=average_embeddings):
#     '''
#     Generate embeddings for an input sequence (in our case, a paragraph)
#     '''
#     embeddings = []
#     count = 0
#     broke_words = []

#     # iterate through the words and get the embedding for each one then apply the averaging function to all
#     for word in words:
#         embedding = embeddings_index.get(word)
#         if embedding is not None:
#             embeddings.append(embedding)
#         else:
#             broke_words.append(word)
#             count += 1
#     return averaging_function(embeddings), count, broke_words  

def save_embeddings(document_embeddings, file_path):
    """
    Save document embeddings to a .npy file.
    """
    # Convert the list to a NumPy array if it's not already
    embeddings_array = np.array(document_embeddings)
    np.save(file_path, embeddings_array)
    logger.info(f"Embeddings saved to {file_path}")

def get_document_embedding_tfidf(words, embeddings_index, word_scores):
    '''
    Generate embeddings for a document using TF-IDF-weighted averaging of word embeddings.

    '''
    embeddings = []
    weights = []
    count = 0
    broke_words = []

    # Get the embedding for easch word
    for word in words:
        embedding = embeddings_index.get(word)
        tfidf_score = word_scores.get(word)
        if embedding is not None and tfidf_score is not None:
            embeddings.append(embedding)
            weights.append(tfidf_score)
        elif embedding is None:
            broke_words.append(word)
            count += 1
            continue
    if embeddings:
        embeddings = np.array(embeddings)
        weights = np.array(weights).reshape(-1, 1)
        weighted_embeddings = embeddings * weights

        #Average weigh the words based on their tfidf weight
        avg_embedding = np.sum(weighted_embeddings, axis=0) / np.sum(weights)
        return avg_embedding, count
    else:
        # Return zero vector if no embeddings found
        print("WTF")
        return np.zeros(300), count
def get_document_embedding(ei):
    # Get the embedding for easch word
    def f(words):
        arr = ['test', 'test2']
        try:
            arr = [str(x) for x in ei]
        except Exception as e:
            logger.error(e)
        finally:
            return arr
    return udf(f)
    
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
        self.conf = SparkConf().setMaster("local[*]").setAppName("SparkTFIDF") \
            .set('spark.local.dir', '/media/volume/team11data/tmp') \
            .set('spark.driver.memory', '50G') \
            .set('spark.driver.maxResultSize', '25G') \
            .set('spark.executor.memory', '10G')
    
        self.sc = SparkContext(conf=self.conf)
        self.sc.setLogLevel("ERROR")
        # sc.setLogLevel("ERROR")
        self.spark = SparkSession(self.sc)
        # self.data_set_rdd = self.spark.createDataFrame(self.data_set).repartition(128, "author_id")
        # self.data_set_rdd = self.spark.createDataFrame(self.data_set)
        self.data_set_rdd = self.spark.read.parquet(self.data_set_path).sort(['author_id', 'book_id'])
        self.data_set_rdd = self.data_set_rdd.withColumn("words", split(self.data_set_rdd.text, " "))
    def stop_spark(self):
        if self.spark is not None:
            self.spark.stop()    

    def extract_ngram_tfidf_features(self):
        '''
        extract_ngram_tfidf_features() will create 'all_data.csv', 'all_labels.csv', and 'all_features.csv' files.
        'all_data.csv': Contains all the data.csv files. Size (237, 7).
        'all_features.csv': All the input features. Size (237, 1000).
        'all_labels.csv': Corresponding author labels (ground truth labels). 1 for "maurice_leblanc" and 0 for others. Size (237, 1).
        '''
        if self.are_features_updated:
            # don't rerun this, just load up datataset
            logger.info("Features are up to date, skipping TF-IDF feature creation")
            return pd.read_parquet(f'{self.data_dir}/tfidf_features.parquet')
        if self.spark is None:
            self.start_spark()
        logger.info("Extracting TF-IDF features...")
        toptimer = timer()
        starttime = timer()
        res=None
        try:
            logger.info(f"Time to startup spark {timer() - starttime}")
            starttime = timer()
           # HashingTF can also be used to get term frequency vectors
            hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
            featurizedData = hashingTF.transform(self.data_set_rdd)
        
            # countVectors = CountVectorizer(inputCol="words", outputCol="rawFeatures", vocabSize=100000, minDF=5)
            # model = countVectors.fit(wordsData)
            # result = model.transform(wordsData)
            logger.info(f"Time to CountVectorizer {timer() - starttime}")

            starttime = timer()
            idf = IDF(inputCol="rawFeatures", outputCol="features")
            idfModel = idf.fit(featurizedData)
            rescaledData = idfModel.transform(featurizedData)
            logger.info(f"Time to IDF {timer() - starttime}")

        
            tfidf_features_df = rescaledData.select("author_id", vector_to_array("features", 'float32').alias("features"))

            if self.labels_arr is not None:
                def is_label_udf(l):
                    def f(x, l):
                        res = None
                        try:
                            res = l[x-1]
                        except Exception as e:
                            logger.info(x)
                            logger.error(e)
                            raise e
                        return res
                    return udf(lambda x: f(x, l), BooleanType())
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
        if not os.path.exists(f"./glove_embeddings.parquet"):
            logger.info("Creating Parquet version of GloVe Embeddings")
            ensure_glove_embeddings(glove_dir='./', glove_file=glove_file_path)
            embeddings_index = load_glove_embeddings(glove_file_path)
            embed_list = [Row(word=k, embedding=v) for k, v in embeddings_index.items()]
            schema = StructType([
                StructField('word', StringType(), True),
                StructField('embedding', VectorUDT(), True)
            ])
            embed_df = self.spark.createDataFrame(embed_list) 
            embed_df.write.parquet("./glove_embeddings.parquet")
            logger.info("Saved Parquet version of GloVe Embeddings to disk")
        else:
            embed_df = self.spark.read.parquet("./glove_embeddings.parquet")
            logger.info("Loaded embedding dataset")
        vectors = []
        num_not_in_vocab = 0
        all_broke_words = [] 
        
        # df = self.data_set_rdd.select('author_id', 'book_id', 'words').withColumn('id', concat(col('author_id').cast("string"),lit('_'), col('book_id').cast("string")))
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
        # e.show(1)
        final_df = df.join(scaled_e, df.id == scaled_e.e_id, "inner")
            
        if self.labels_arr is not None:
            def is_label_udf(l):
                def f(x, l):
                    res = None
                    try:
                        res = l[x-1]
                    except Exception as e:
                        logger.info(x)
                        logger.error(e)
                        raise e
                    return res
                return udf(lambda x: f(x, l), BooleanType())
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
