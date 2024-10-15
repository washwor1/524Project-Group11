import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import glob

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
            df['author'] =  data_file.split('\\')[-3] 
            df['labels'] = (df['author'] == 'maurice_leblanc').astype(int)
            data_sets.append(df)
    
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



def extract_features(data_dir='data'):
    fean = Feature_analysis(data_dir)
    fean.extract_ngram_tfidf_features()

if __name__ == "__main__":
    extract_features()