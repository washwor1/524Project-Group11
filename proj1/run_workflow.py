import get_novels
import preprocess_data
from feature_engineering import Feature_analysis

# Step 1: Get the novels from the database, they are saved to data_dir (default './data')
get_novels.get_novels_wrapper()

# Step 2: Break all novels into individual sections and store them in individualized csv
preprocess_data.process_all_files()

# Step 3: Get the features from all the data we got from Step 2
Feature_analysis.extract_ngram_tfidf_features()

