from preprocess_data import process_all_files
from feature_engineering import extract_features
from modeling import model_data
import os

# Set the data directory
data_dir = f"{os.path.dirname(__file__)}/data"

# Function to run a single configuration
def run_single_configuration(settings):
    # Unpack settings
    group_by_paragraph = settings[0]
    remove_stopwords = settings[1]
    keep_punctuation = settings[2]
    group_by_length = settings[3]
    group_length = settings[4]
    multiclass = settings[5]
    remove_out_of_vocab = settings[6]
    
    # Process all files with the specified settings
    process_all_files(
        data_dir,
        group_by_paragraphs=group_by_paragraph,
        remove_stopword=remove_stopwords,
        keep_punctuations=keep_punctuation,
        group_by_lengths=group_by_length,
        group_lengths=group_length
    )
    
    # Extract features with the specified settings
    extract_features(
        data_dir,
        multiclass_classification=multiclass,
        remove_out_of_vocabs=remove_out_of_vocab
    )
    
    # Run the model and log the results
    model_data(settings, metrics_file="metrics_single_run.csv")

# Settings order:
# settings_info[0] = group_by_paragraph
# settings_info[1] = remove_stopwords
# settings_info[2] = keep_punctuation
# settings_info[3] = group_by_length
# settings_info[4] = group_length
# settings_info[5] = multiclass
# settings_info[6] = remove_out_of_vocab

settings = [False,True,True,True,300,False,False]
run_single_configuration(settings)

settings = [False,False,True,True,300,False,False]
run_single_configuration(settings)