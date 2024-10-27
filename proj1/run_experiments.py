from preprocess_data import process_all_files
from feature_engineering import extract_features
from modeling import model_data
import os

file_path = 'metrics.csv'
if os.path.exists(file_path):
    try:
        os.remove(file_path)
        print(f"{file_path} has been deleted.")
    except OSError as e:
        print(f"Error deleting {file_path}: {e}")
else:
    print(f"{file_path} does not exist.")

data_dir = f"{os.path.dirname(__file__)}/data"

embeddings_index = None
# Experiment Parameters
# Available:
#     Preprocess:
#         - groupby: paragraph, length, None
#             - groupby length: scalar (0-~900)
#         - remove_stopwords [true/false]
#         - keep_punctuation [true/false]

#     Feature Engineering:
#         - vectorization type: tfidf, glove, glove-tfidf (Will be handled by the modeling program. Automatically does all 3)
#         - remove out of vocab: [true/false]
#         - Classification Type: [binary/multiclass]

# Default settings tuple
default_settings = [True, False, False, False, None, False, False]

# Parameters and their indices in the settings tuple
params = [
    ('group_by_paragraph', 0),
    ('remove_stopwords', 1),
    ('keep_punctuation', 2),
    ('group_by_length', 3),
    ('multiclass', 5),
    ('remove_out_of_vocab', 6)
]

group_lengths = [100, 300, 500]

# Run with default settings (group_by_paragraph=True)
settings = tuple(default_settings)
process_all_files(
    data_dir,
    group_by_paragraphs=settings[0],
    remove_stopword=settings[1],
    keep_punctuations=settings[2],
    group_by_lengths=settings[3],
    group_lengths=settings[4]
)
embeddings_index = extract_features(
    data_dir,
    multiclass_classification=settings[5],
    remove_out_of_vocabs=settings[6],
    embeddings_index=embeddings_index
)
model_data(settings)

# Run a single execution where group_by_paragraph is False
settings_list = default_settings.copy()
settings_list[0] = False  # Set group_by_paragraph to False
settings = tuple(settings_list)
process_all_files(
    data_dir,
    group_by_paragraphs=settings[0],
    remove_stopword=settings[1],
    keep_punctuations=settings[2],
    group_by_lengths=settings[3],
    group_lengths=settings[4]
)
extract_features(
    data_dir,
    multiclass_classification=settings[5],
    remove_out_of_vocabs=settings[6],
    embeddings_index=embeddings_index
)
model_data(settings)

# Iterate over each parameter except 'group_by_paragraph'
for param_name, index in params:
    if param_name == 'group_by_paragraph':
        continue  # Already handled
    settings_list = default_settings.copy()
    settings_list[index] = not default_settings[index]

    # Handle group lengths
    if param_name == 'group_by_length':
        settings_list[index] = True
        for gl in group_lengths:
            settings_list[4] = gl
            settings = tuple(settings_list)
            process_all_files(
                data_dir,
                group_by_paragraphs=settings[0],  # Default True
                remove_stopword=settings[1],
                keep_punctuations=settings[2],
                group_by_lengths=settings[3],
                group_lengths=settings[4]
            )
            extract_features(
                data_dir,
                multiclass_classification=settings[5],
                remove_out_of_vocabs=settings[6],
                embeddings_index=embeddings_index
            )
            model_data(settings)
    else:
        settings_list[4] = None
        settings = tuple(settings_list)
        process_all_files(
            data_dir,
            group_by_paragraphs=settings[0],  # Default True
            remove_stopword=settings[1],
            keep_punctuations=settings[2],
            group_by_lengths=settings[3],
            group_lengths=settings[4]
        )
        extract_features(
            data_dir,
            multiclass_classification=settings[5],
            remove_out_of_vocabs=settings[6],
            embeddings_index=embeddings_index
        )
        model_data(settings)

# Run with best settings
settings = [True, False, True, True, 300, False, False]
process_all_files(
    data_dir,
    group_by_paragraphs=settings[0],
    remove_stopword=settings[1],
    keep_punctuations=settings[2],
    group_by_lengths=settings[3],
    group_lengths=settings[4]
)
extract_features(
    data_dir,
    multiclass_classification=settings[5],
    remove_out_of_vocabs=settings[6],
    embeddings_index=embeddings_index
)
model_data(settings)
