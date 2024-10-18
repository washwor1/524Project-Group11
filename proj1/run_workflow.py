from get_novels import get_novels
from preprocess_data import process_all_files
from feature_engineering import extract_features
import os


data_dir = f"{os.path.dirname(__file__)}\data"
# Step 1: Get the novels from the database, they are saved to data_dir (default './data')
get_novels(data_dir)

# Step 2: Break all novels into individual sections and store them in individualized csv
process_all_files(data_dir)

# Step 3: Get the features from all the data we got from Step 2
extract_features(data_dir)

