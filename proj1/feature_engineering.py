import os
import nltk
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import string


class Feature_analysis():
    def __init__(self):
        self.directory = r"proj1\data"
        
        self.data_set = None
        self.ngram_range = (1, 2) #we are using unigram and bigram
        self.max_features = 100  #number of features we want from teh dataset as inputs for the model

        all_data_file = 'all_data.csv'
        file_path = os.path.join(self.directory, all_data_file)

        # Check if the file exists
        if os.path.isfile(file_path):
            self.data_set = pd.read_csv(file_path)
        else:
            self.load_dataset()
            self.save_dataset()


    # Load the dataset
    def load_dataset(self):
        data_sets = []
        for author_dir in os.listdir(self.directory):
            author_path = os.path.join(self.directory, author_dir)
            all_files = os.listdir(author_path)
            book_files = [f for f in all_files if not f.endswith('.txt')]

            for filename in book_files:
                filepath = os.path.join(author_path, filename)
                all_data = os.listdir(filepath)
                data_files = [f for f in all_data if f.endswith('.csv')]

                for data in data_files:
                    data_path = os.path.join(filepath, data)
                    df = pd.read_csv(data_path)
                    df['author'] = author_dir
                    df['labels'] = (df['author'] == 'maurice_leblanc').astype(int)
                    data_sets.append(df)
    
        self.data_set = pd.concat(data_sets) 
        self.data_set.reset_index(drop=True, inplace=True)
    
    def save_dataset(self):
        self.data_set.to_csv(f'{self.directory}/all_data.csv', index=False)
        self.data_set['labels'].to_frame().to_csv(f'{self.directory}/all_labels.csv', index=False)

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
        tokens = [token for token in tokens if token not in string.punctuation]  # Remove punctuation
        tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
        return ' '.join(tokens)
    
    def extract_ngram_tfidf_features(self):
        '''
         extract_ngram_tfidf_features() will create 'all_data.csv', 'all_labels.csv" and 'all_features.csv' files.
        'all_data.csv': It consists all the data.csv files. size (237, 7),  
        'all_features.csv': all the input features. (237, 1000), 
        'all_labels.csv": corresponding author labels (ground truth labels). 1 for "maurice_leblanc" and 0 for others. size (237, 1)

        '''

        self.data_set['clean_text'] = self.data_set['text'].astype(str)
        self.data_set['clean_text'] = self.data_set['clean_text'].apply(self.preprocess_text)

        tfidf_vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        tfidf_features = tfidf_vectorizer.fit_transform(self.data_set['clean_text'])
        tfidf_features_df = pd.DataFrame(tfidf_features.toarray())
        tfidf_features_df.to_csv(f'{self.directory}/all_features.csv', index=False)


fean = Feature_analysis()
fean.extract_ngram_tfidf_features()
