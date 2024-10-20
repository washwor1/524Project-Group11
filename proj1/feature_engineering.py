import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import glob
import os
import requests
import zipfile

def load_glove_embeddings(glove_file_path):
    '''
    Loads the glove embeddings from the .txt file into memory. Takes a while and needs 2gb+ memory
    '''
    embeddings_index = {}
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

    I made averaging function a parameter because apparenlly it is possible to use tfidf on these
    '''
    embeddings = []
    count = 0
    broke_words = []
    for word in words:
        embedding = embeddings_index.get(word)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            broke_words.append(word)
            count+=1
            continue  # Skip 
    with open('thrown_out_words.txt', 'a+') as f:
        for line in broke_words:
            f.write(f"{line}\n")
    return averaging_function(embeddings), count

def save_embeddings(document_embeddings, file_path):
    """
    Save document embeddings to a .npy file.
    """
    # Convert the list to a NumPy array if it's not already
    embeddings_array = np.array(document_embeddings)
    np.save(file_path, embeddings_array)
    print(f"Embeddings saved to {file_path}")



class Feature_analysis():
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        
        self.data_set = None
        self.ngram_range = (1, 2) #we are using unigram and bigram
        self.max_features = 100  #number of features we want from teh dataset as inputs for the model

        # Check if the file exists
        # file_path = os.path.join(self.data_dir, 'all_data.csv')
        # if os.path.isfile(file_path):
        #     self.data_set = pd.read_csv(file_path)
        # else:
        self.load_dataset()
        self.save_dataset()


    def load_dataset(self):
        data_sets = []
        data_files = glob.glob(f"{self.data_dir}/**/**/**.csv")
        for data_file in data_files:
            df = pd.read_csv(data_file)
            df = df.dropna()
            df['author'] =  data_file.split('/')[-3] 
            df['labels'] = (df['author'] == 'maurice_leblanc').astype(int)
            df['text'] = df['text'].str.split('|')
            df_exploded = df.explode('text')
            data_sets.append(df_exploded)
    
        self.data_set = pd.concat(data_sets) 
        self.data_set.reset_index(drop=True, inplace=True)
    
    def save_dataset(self):
        self.data_set.to_csv(f'{self.data_dir}/all_data.csv', index=False)
        self.data_set['labels'].to_frame().to_csv(f'{self.data_dir}/all_labels.csv', index=False)

    def extract_ngram_tfidf_features(self):
        '''
         extract_ngram_tfidf_features() will create 'all_data.csv', 'all_labels.csv" and 'all_features.csv' files.
        'all_data.csv': It consists all the data.csv files. size (237, 7),  
        'all_features.csv': all the input features. (237, 1000), 
        'all_labels.csv": corresponding author labels (ground truth labels). 1 for "maurice_leblanc" and 0 for others. size (237, 1)

        '''
        print("Extracting TF-IDF features...")
        tfidf_vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        tfidf_features = tfidf_vectorizer.fit_transform(self.data_set['text'])
        arr = tfidf_vectorizer.get_feature_names_out()
        with open(f"{self.data_dir}/array.txt", 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(arr))
        tfidf_features_df = pd.DataFrame(tfidf_features.toarray(), columns=arr)
        print("Saving features...")
        # tfidf_features_df['name'] = tfidf_vectorizer.get_feature_names_out()
        tfidf_features_df.to_csv(f'{self.data_dir}/all_features.csv', index=False)
        print(f"Saved features to {self.data_dir}/all_features.csv")
    
    def generate_glove_vecs(self):
        '''
        Generates the glove vectors for each chapter in the dataset. 

        Saves them to a numpy array file 'document_embeddings.npy'
        '''
        glove_file_path = 'glove.840B.300d.txt'

        # WILL DOWNLOAD 2GB FILE
        ensure_glove_embeddings(glove_dir='./', glove_file=glove_file_path)
        embeddings_index = load_glove_embeddings(glove_file_path)

        vectors = []
        num_not_in_vocab = 0
        for text in self.data_set['text']:
            single_vec, num = get_document_embedding(text.strip().split(' '), embeddings_index)
            num_not_in_vocab += num
            vectors.append(single_vec)
        
        num_docs = (len(self.data_set['text']))
        print(f'Average Number of Words not in Embedding Vocab: {num_not_in_vocab/num_docs}')
        save_embeddings(vectors, 'document_embeddings.npy')





def extract_features(data_dir='data'):
    fean = Feature_analysis(data_dir)

    # IF YOU DONT HAVE THE GLOVE EMBEDDINGS, WILL DOWNLOAD 2GB FILE.
    fean.generate_glove_vecs()
    
    # fean.extract_ngram_tfidf_features()

if __name__ == "__main__":
    extract_features()
