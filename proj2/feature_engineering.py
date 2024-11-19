'''
    Contains functions and code to generate TF-IDF and GloVe word embeddings
    (adapted from Project 1 for this course)
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import glob
import os
import requests
import zipfile
import time
import logging

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
                embedding = np.asarray(values[1:], dtype='float32')
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
        print("GloVe embeddings not found. Downloading...")

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
        print("\nDownload complete. Extracting...")

        # Extract the .txt file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(glove_dir)
        print("Extraction complete.")

        # Delete the zip file
        os.remove(zip_path)
        print("Zip file deleted.")
    else:
        print("GloVe embeddings already exist.")

def average_embeddings(embeddings):
    '''
    Simple function to average glove embeddings for a sequence
    '''
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        #Returns zeros if the embeddings are empty for some reason. 
        return np.zeros(300)  

def get_document_embedding(words, embeddings_index, averaging_function=average_embeddings):
    '''
    Generate embeddings for an input sequence (in our case, a paragraph)
    '''
    embeddings = []
    count = 0
    broke_words = []

    # iterate through the words and get the embedding for each one then apply the averaging function to all
    for word in words:
        embedding = embeddings_index.get(word)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            broke_words.append(word)
            count += 1
    return averaging_function(embeddings), count, broke_words  

def save_embeddings(document_embeddings, file_path):
    """
    Save document embeddings to a .npy file.
    """
    # Convert the list to a NumPy array if it's not already
    embeddings_array = np.array(document_embeddings)
    np.save(file_path, embeddings_array)
    print(f"Embeddings saved to {file_path}")

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

class FeatureAnalysis():
    '''
    Class used to manage the feature engineering data
    '''
    def __init__(self, df):
        self.data_dir = "data"
        
        self.data_set = df
        self.ngram_range = (1, 2) #we are using unigram and bigram
        self.max_features = 100  #number of features we want from teh dataset as inputs for the model

        # self.load_dataset()
        # self.save_dataset()

    def extract_ngram_tfidf_features(self):
        '''
        extract_ngram_tfidf_features() will create 'all_data.csv', 'all_labels.csv', and 'all_features.csv' files.
        'all_data.csv': Contains all the data.csv files. Size (237, 7).
        'all_features.csv': All the input features. Size (237, 1000).
        'all_labels.csv': Corresponding author labels (ground truth labels). 1 for "maurice_leblanc" and 0 for others. Size (237, 1).
        '''
        print("Extracting TF-IDF features...")

        
        # Set up tfidf vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range
        )
        tfidf_features = tfidf_vectorizer.fit_transform(self.data_set['text'])
        feature_names = tfidf_vectorizer.get_feature_names_out()

        tfidf_features_df = pd.DataFrame(tfidf_features.toarray(), columns=feature_names)

        return tfidf_features_df


    
    def generate_glove_vecs(self, embeddings_index=None):
        '''
        Generates the glove vectors for each chapter in the dataset. 

        Saves them to a numpy array file 'document_embeddings.npy'
        '''
        glove_file_path = 'glove.840B.300d.txt'

        # WILL DOWNLOAD 2GB FILE
        if embeddings_index is None:
            ensure_glove_embeddings(glove_dir='./', glove_file=glove_file_path)
            embeddings_index = load_glove_embeddings(glove_file_path)

        vectors = []
        num_not_in_vocab = 0
        all_broke_words = []  

        # iterate through the different texts and get the embed for all of them
        # collects the words that didnt have an embedding for analysis 
        for text in self.data_set['text']:
            single_vec, num, broke_words = get_document_embedding(text.strip().split(' '), embeddings_index)
            num_not_in_vocab += num
            vectors.append(single_vec)
            all_broke_words.extend(broke_words) 
        
        num_docs = len(self.data_set['text'])
        print(f'Average Number of Words not in Embedding Vocab: {num_not_in_vocab/num_docs}')
        save_embeddings(vectors, 'document_embeddings.npy')

        # Write all words without an embedding to the file
        with open('thrown_out_words.txt', 'w', encoding='utf-8') as f:
            for word in all_broke_words:
                f.write(f"{word}\n")

        return embeddings_index
    
    def generate_glove_vecs_with_tfidf(self, embeddings_index=None):
        '''
        Generates the GloVe vectors for each chapter in the dataset, weighted by TF-IDF scores.

        Saves them to a numpy array file 'document_embeddings_tfidf.npy'.
        '''
        glove_file_path = 'glove.840B.300d.txt'

        # load the embeddings if necessary
        if embeddings_index is None:
            ensure_glove_embeddings(glove_dir='./', glove_file=glove_file_path)
            embeddings_index = load_glove_embeddings(glove_file_path)

        print("Computing TF-IDF scores...")

        # generate tfidf vector 
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1,3)
        )

        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data_set['text'])
        feature_names = tfidf_vectorizer.get_feature_names_out()

        vectors = []
        num_not_in_vocab = 0

        # Get the analyzer function from the vectorizer to ensure consistent tokenization
        analyzer = tfidf_vectorizer.build_analyzer()

        for doc_index, text in enumerate(self.data_set['text']):
            # Use the analyzer to get tokens, ensuring consistency with TF-IDF vectorizer
            words = analyzer(text)
            tfidf_vector = tfidf_matrix[doc_index]
            coo = tfidf_vector.tocoo()
            word_scores = {}
            for idx, value in zip(coo.col, coo.data):
                word = feature_names[idx]
                word_scores[word] = value

            embedding, num = get_document_embedding_tfidf(words, embeddings_index, word_scores)

            num_not_in_vocab += num
            vectors.append(embedding)

        num_docs = len(self.data_set['text'])
        print(f'Average Number of Words not in Embedding Vocab: {num_not_in_vocab / num_docs}')
        save_embeddings(vectors, 'document_embeddings_tfidf.npy')

        return vectors



def extract_features(df, multiclass_classification = False, remove_out_of_vocabs = False, embeddings_index=None):
    global multiclass
    multiclass=multiclass_classification
    
    global remove_out_of_vocab
    remove_out_of_vocab=remove_out_of_vocabs

    fean = FeatureAnalysis(df)
    logger = logging.getLogger("proj2_logger")
    start_time = time.time()
    # IF YOU DONT HAVE THE GLOVE EMBEDDINGS, WILL DOWNLOAD 2GB FILE.
    # embeddings_index = fean.generate_glove_vecs(embeddings_index)
    # vecs = fean.generate_glove_vecs_with_tfidf(embeddings_index)
    logger.info(f"Finished getting word embeddings (took {time.time() - start_time})")
    tfidf_features = fean.extract_ngram_tfidf_features()
    end_time = time.time()
    tfidf_features.to_csv("all_tfdf_features.csv", index=False)
    logger.info(f"Finished extracting features (took {(end_time - start_time)})")
    
    # Return embeddings index so they can be used in the UI
    return tfidf_features
