import numpy as np 
import pandas as pd
from contextlib import redirect_stdout

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import tensorflow as tf
import os

def gaussian(input_features, input_labels):
    '''
    Runs Gaussian Naive Bayes model 
    '''
    X = np.array(input_features)
    y = input_labels['labels']  
    print(f'Sanity Check\nNumber of embeddings loaded: {len(X)}\nNumber of matching labels: {len(y)}\n')
    clf = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = [accuracy_score(y_test, y_pred), *[np.mean(x) for x in precision_recall_fscore_support(y_test, y_pred, zero_division=0)]]
    return(metrics, classification_report(y_test, y_pred, zero_division=0))

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
    return(metrics, classification_report(y_test, y_pred, zero_division=0))

def lstm(input_features, input_labels):
    '''
    Runs a LSTM model 
    '''
    Xtrain_lstm = input_features
    Xtrain_lstm = np.array(input_features).reshape((Xtrain_lstm.shape[0], 1, Xtrain_lstm.shape[1]))
    lstm = Sequential()
    lstm.add(Input(shape=(Xtrain_lstm.shape[1], Xtrain_lstm.shape[2])))
    lstm.add(LSTM(units = 128))
    lstm.add(Dense(1, activation='sigmoid'))
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Accuracy(),tf.keras.metrics.F1Score(),tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    lstm.summary()
    X_train, X_test, y_train, y_test = train_test_split(Xtrain_lstm, input_labels, train_size= 0.8, random_state=42)
    print(X_train.shape)
    print(y_train.shape)
    lstm.fit(X_train, y_train, verbose = 0, epochs=10, batch_size=32, validation_split=0.2)
    loss, accuracy, f1_score, precision, recall = lstm.evaluate(X_test, y_test, verbose=0)
    y_pred = lstm.predict(X_test, batch_size=64, verbose=0)
    y_pred_bool = np.argmax(y_pred, axis=1)

    metrics = [accuracy, precision, recall, f1_score, None]
    return (metrics, classification_report(y_test, y_pred_bool, zero_division=0))

def rf(input_features, input_labels):
    X = np.array(input_features)
    y = input_labels['labels']  
    print(f'Sanity Check\nNumber of embeddings loaded: {len(X)}\nNumber of matching labels: {len(y)}\n')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = [accuracy_score(y_test, y_pred), *[np.mean(x) for x in precision_recall_fscore_support(y_test, y_pred, zero_division=0)]]
    return(metrics, classification_report(y_test, y_pred, zero_division=0))

def load_embeddings(file_path):
    """
    Load document embeddings from a .npy file.
    """
    embeddings_array = np.load(file_path)
    print(f"Embeddings loaded from {file_path}")
    return embeddings_array

def model_data():

    # Get the current working directory
    current_dir = os.getcwd()

    # Assuming the CSV files are located in a 'data' folder within the project directory
    tfidf_features_path = os.path.join(current_dir, 'data', 'all_features.csv')
    tfidf_features = pd.read_csv(tfidf_features_path)

    labels_path = os.path.join(current_dir, 'data', 'all_labels.csv')

    # Load the data
    Labels = pd.read_csv(labels_path)

    # Load the extracted feature embeddings from glove
    loaded_embeddings = load_embeddings('document_embeddings_tfidf.npy')


    functions = [rf, gaussian, svm, lstm]
    metrics_arr = []
    os.makedirs("modelTesting", exist_ok=True)
    for feature_type in ['glove', 'tfidf']:
        print(f"{'-' * 25} Began testing {feature_type} Features {'-' * 25}")
        features = loaded_embeddings if feature_type == 'glove' else tfidf_features

        for function in functions:
            with open(f'modelTesting/{function.__name__}-{feature_type}.txt', 'w', encoding='utf-8') as f:
                with redirect_stdout(f):
                    print("Results: ")
                    try:
                        metrics, classification_report = function(features, Labels)
                    except Exception as e:
                        print(e)
                    metrics_arr.append([function.__name__, feature_type, *metrics])
                    print(metrics)
                    print("--------")
                    print(classification_report)
            print(f"----- Tested {function.__name__} model")
        print(f"{'-' * 25} Finished testing {feature_type} Features {'-' * 25}")

    df = pd.DataFrame(metrics_arr, columns=['model', 'embedding_type', 'accuracy', 'precision', 'recall', 'f1-score', 'support'])
    df.to_csv("metrics.csv", index=False)

if __name__ == "__main__":
    model_data()
