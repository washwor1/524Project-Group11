import numpy as np 
import pandas as pd
from contextlib import redirect_stdout

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support ,precision_score, recall_score, f1_score
from sklearn.svm import SVC
import re
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import tensorflow as tf
import os
import glob
import nltk
from nltk.corpus import stopwords
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess_data import process_all_files
from feature_engineering import extract_features

def svm(input_features, input_labels):
    '''
    Runs Support Vector Classifier (a type of Support Vector Machine)
    '''
    X = np.array(input_features)
    y = input_labels['labels']  
    print(f'Sanity Check\nNumber of embeddings loaded: {len(X)}\nNumber of matching labels: {len(y)}\n')
    clf = SVC(kernel = "sigmoid") # try different kernel
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = [accuracy_score(y_test, y_pred), *[np.mean(x) for x in precision_recall_fscore_support(y_test, y_pred, zero_division=0)]]
    return(metrics, classification_report(y_test, y_pred, zero_division=0), clf)

def lstm(input_features, input_labels):
    '''
    Runs an LSTM model
    '''
    # Determine the number of unique classes
    classes = np.unique(input_labels)
    num_classes = len(classes)

    # Check if the problem is binary or multiclass
    is_binary = num_classes == 2

    # Prepare input data
    Xtrain_lstm = np.array(input_features)
    Xtrain_lstm = Xtrain_lstm.reshape((Xtrain_lstm.shape[0], 1, Xtrain_lstm.shape[1]))

    # Build the LSTM model
    lstm = Sequential()
    lstm.add(Input(shape=(Xtrain_lstm.shape[1], Xtrain_lstm.shape[2])))
    lstm.add(LSTM(units=128))

    if is_binary:
        # Binary classification configuration
        lstm.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    else:
        # Multiclass classification configuration
        lstm.add(Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'  
        metrics = ['accuracy']

    lstm.compile(
        optimizer='adam',
        loss=loss,
        metrics=metrics
    )
    lstm.summary()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        Xtrain_lstm, input_labels, train_size=0.8, random_state=42
    )
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Train the model
    lstm.fit(
        X_train, y_train,
        verbose=0,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # Make predictions
    y_pred = lstm.predict(X_test, batch_size=64, verbose=0)

    if is_binary:
        # For binary classification, threshold the probabilities
        y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, zero_division=0)
        recall = recall_score(y_test, y_pred_classes, zero_division=0)
        f1 = f1_score(y_test, y_pred_classes, zero_division=0)
        metrics_list = [accuracy, precision, recall, f1, None]
        # Generate classification report
        class_report = classification_report(y_test, y_pred_classes, zero_division=0)
    else:
        # For multiclass classification, select the class with highest probability
        y_pred_classes = np.argmax(y_pred, axis=1)
        # Compute metrics with appropriate averaging method
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred_classes, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred_classes, average='macro', zero_division=0)
        metrics_list = [accuracy, precision, recall, f1, None]
        # Generate classification report
        class_report = classification_report(y_test, y_pred_classes, zero_division=0)

    print(metrics_list)
    # Return metrics and classification report
    return (metrics_list, class_report, lstm)


def load_embeddings(file_path):
    """
    Load document embeddings from a .npy file.
    """
    embeddings_array = np.load(file_path)
    print(f"Embeddings loaded from {file_path}")
    return embeddings_array

def model_data(settings_info):
    group_by_paragraph = settings_info[0] 
    remove_stopwords = settings_info[1]
    keep_punctuation = settings_info[2] 
    group_by_length = settings_info[3] 
    group_length = settings_info[4]  
    multiclass = settings_info[5]
    remove_out_of_vocab = settings_info[6]

    # Get the current working directory
    current_dir = os.getcwd()

    # Assuming the CSV files are located in a 'data' folder within the project directory
    tfidf_features_path = os.path.join(current_dir, 'data', 'all_features.csv')
    tfidf_features = pd.read_csv(tfidf_features_path)

    labels_path = os.path.join(current_dir, 'data', 'all_labels.csv')

    # Load the data
    Labels = pd.read_csv(labels_path)

    # Load the extracted feature embeddings from glove
    loaded_embeddings = load_embeddings('document_embeddings.npy')
    loaded_embeddings_tfidf = load_embeddings('document_embeddings_tfidf.npy')



    functions = [svm, lstm]
    metrics_arr = []
    models=[]
    os.makedirs("modelTesting", exist_ok=True)
    for feature_type in ['glove', 'tfidf']:
        print(f"{'-' * 25} Began testing {feature_type} Features {'-' * 25}")
        if feature_type == 'glove':
            features = loaded_embeddings
        elif feature_type == 'glove-tfidf':
            features = loaded_embeddings_tfidf
        else:
            features = tfidf_features

        for function in functions:
            with open(f'modelTesting/{function.__name__}-{feature_type}.txt', 'w', encoding='utf-8') as f:
                with redirect_stdout(f):
                    print("Results: ")
                    try:
                        metrics, classification_report, model = function(features, Labels)
                        models.append(model)
                    except Exception as e:
                        print(e)
                    metrics_arr.append([function.__name__, feature_type, *metrics, group_by_paragraph, remove_stopwords, keep_punctuation, group_by_length, group_length, multiclass, remove_out_of_vocab])
                    print(metrics)
                    print("--------")
                    print(classification_report)
            print(f"----- Tested {function.__name__} model")
        print(f"{'-' * 25} Finished testing {feature_type} Features {'-' * 25}")
    
    return models




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
    Compact document embeddings for user input.
    '''
    embeddings = []
    for word in words:
        embedding = embeddings_index.get(word)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            continue  
    return averaging_function(embeddings)   

def extract_ngram_tfidf_features(data_set, vectorizer):
    # Set up tfidf vectorizer
    tfidf_vectorizer = vectorizer
    tfidf_features = tfidf_vectorizer.transform(data_set)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return pd.DataFrame(tfidf_features.toarray(), columns=feature_names)

def normalize_text_block_compact(block: str):
    '''
    Apply various patterns to normalize a block of text
    Compact version that ignores the options in the preprocess_data.py
    '''
    normalizing_patterns = {
        r"^\s*[\s*]+$": "",  # filter out line breaks with asterisk
        r"[“”]": "\"",       # replace Unicode quotes with standard ones
        r"[‘’]": "'",        # replace Unicode apostrophes with standard ones
        r"[–—]": "-",        # replace long dashes with standard dash
        r"\[((.|\n|\r)*?)\]$": "",  # remove notes at the end
        r"\[((.|\n|\r)*?)\]": "",   # remove inline notes
        r"\|(.*)\|": "",     # remove text between bars
        r"\s+\+-+\+": "",    # remove ASCII art headers
        r"^\s+\.{5,}((.|\n|\r)*?)\.{5,}.*$": "",  # remove patterns of dots
        r"\n{3,}": "\n\n",   # shorten large margins
        r"\"": "",           # remove quotes
        r"--": "",           # remove double dashes
        r"^End of Project Gutenberg's .*$": "",  # remove specific ending line
        r"(\d{4,}|\d{1,3}-\d{1,3}-\d{1,3})": "", # remove years and long numbers
        r"\d{2,}\.\d{2}": "",  # remove times formatted as hh.mm
        r"\d{1,}(th|st|nd|rd)": "",  # remove ordinal numbers
        r"(\$|£)?\s?([0-9]{1,3},)*[0-9]{1,3}": "",  # remove currency amounts
        r"vi{2,}": ""        # remove repeated 'vi'
    }

    # Apply normalization patterns
    for pat, sub in normalizing_patterns.items():
        block = re.sub(pat, sub, block, flags=re.MULTILINE)

    # Strip whitespace and fix contractions
    block = block.strip()
    block = contractions.fix(block)

    # Tokenize and convert to lowercase without removing stopwords or punctuation
    tokens = [token.lower() for token in nltk.word_tokenize(block) if (token not in nltk.corpus.stopwords.words("english"))]

    # Join tokens back into a string
    res = ' '.join(tokens)
    return res


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
        group_by_paragraphs=group_by_paragraph,
        remove_stopword=remove_stopwords,
        keep_punctuations=keep_punctuation,
        group_by_lengths=group_by_length,
        group_lengths=group_length
    )
    
    # Extract features with the specified settings
    embed, vectorizer = extract_features(
        multiclass_classification=multiclass,
        remove_out_of_vocabs=remove_out_of_vocab
    )
    
    # Run the model and log the results
    models = model_data(settings)
    return models, embed, vectorizer

# Settings order:
# settings_info[0] = group_by_paragraph
# settings_info[1] = remove_stopwords
# settings_info[2] = keep_punctuation
# settings_info[3] = group_by_length
# settings_info[4] = group_length
# settings_info[5] = multiclass
# settings_info[6] = remove_out_of_vocab
if __name__ == "__main__":
    # Define the path to your directory containing the text files
    folder_path = 'ind_test'  # Update this path if necessary

    # Prepare lists to store file contents and file names
    file_contents = []
    file_names = []

    # Use glob to find all .txt files in the directory
    file_pattern = os.path.join(folder_path, '*.txt')
    txt_files = glob.glob(file_pattern)

    # Loop through each file and read its content
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            file_contents.append(content)
            file_names.append(os.path.basename(file_path))

    # Define the settings for your configuration
    settings = [False, True, True, True, 500, False, False]
    models, embed, vectorizer = run_single_configuration(settings)

    # Prepare a list to store the results
    results_list = []

    # Process each text and make predictions
    for idx, text in enumerate(file_contents):
        print(f"Processing file: {file_names[idx]}")

        # Normalize the text
        text_normalized = normalize_text_block_compact(text)

        # --- Embedding-based Predictions ---
        # Get the document embedding
        vec = get_document_embedding(text_normalized, embed)
        keras_vec = np.array(vec)

        # Predict using the sklearn model (models[0])
        pred = models[0].predict(keras_vec.reshape(1, -1))

        # Predict using the Keras model (models[1])
        keras_vec_reshaped = keras_vec.reshape(1, 1, -1)
        pred2 = models[1].predict(keras_vec_reshaped)
        pred2_class = (pred2 >= 0.5).astype(int)

        print("Embedding-based predictions:")
        print(f"Model 0 prediction: {pred[0]}")
        print(f"Model 1 prediction: {pred2_class[0][0]}")

        # --- TF-IDF-based Predictions ---
        # Extract TF-IDF features
        vec_tfidf = extract_ngram_tfidf_features([text_normalized], vectorizer)
        keras_vec_tfidf = np.array(vec_tfidf)  # Convert sparse matrix to dense array

        # Predict using the sklearn model (models[2])
        pred_tfidf = models[2].predict(keras_vec_tfidf.reshape(1, -1))

        # Predict using the Keras model (models[3])
        keras_vec_tfidf_reshaped = keras_vec_tfidf.reshape(1, 1, -1)
        pred2_tfidf = models[3].predict(keras_vec_tfidf_reshaped)
        pred2_tfidf_class = (pred2_tfidf >= 0.5).astype(int)

        print("TF-IDF-based predictions:")
        print(f"Model 2 prediction: {pred_tfidf[0]}")
        print(f"Model 3 prediction: {pred2_tfidf_class[0][0]}")
        print("-" * 50)

        # Store the predictions in the results list
        results_list.append({
            'file_name': file_names[idx],
            'pred_model0': pred[0],
            'pred_model1': pred2_class[0][0],
            'pred_model2': pred_tfidf[0],
            'pred_model3': pred2_tfidf_class[0][0]
        })

    # Create a DataFrame to hold the results
    results = pd.DataFrame(results_list)

    # Print or save the results
    print(results)
    # Optionally, save the results to a CSV file
    results.to_csv('predictions_results.csv', index=False)
