'''
    Contains classical ML models for the experiment
    (adapted from Project 1)
'''
import numpy as np 
import pandas as pd
from contextlib import redirect_stdout

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support ,precision_score, recall_score, f1_score, precision_recall_curve
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
    metrics = [accuracy_score(y_test, y_pred), *[np.mean(x) for x in precision_recall_fscore_support(y_test, y_pred, zero_division=0, average='macro')]]
    y_prob_test = clf.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob_test)
    return(metrics, classification_report(y_test, y_pred, zero_division=0), (precision, recall))

def svm(input_features, input_labels):
    '''
    Runs Support Vector Classifier (a type of Support Vector Machine)
    '''
    X = np.array(input_features)
    y = input_labels['labels']  
    print(f'Sanity Check\nNumber of embeddings loaded: {len(X)}\nNumber of matching labels: {len(y)}\n')
    clf = SVC(kernel = "sigmoid", probability=True) # try different kernel
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob_test)
    metrics = [accuracy_score(y_test, y_pred), *[np.mean(x) for x in precision_recall_fscore_support(y_test, y_pred, zero_division=0, average='macro')]]
    return(metrics, classification_report(y_test, y_pred, zero_division=0), (precision, recall))

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
    print(y_pred)
    plots = None
    if is_binary:
        # For binary classification, threshold the probabilities
        y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, zero_division=0)
        recall = recall_score(y_test, y_pred_classes, zero_division=0)
        f1 = f1_score(y_test, y_pred_classes, zero_division=0)
        metrics_list = [accuracy, precision, recall, f1, None]
        p_precision, p_recall, thresholds = precision_recall_curve(y_test, y_pred_classes)
        plots = (p_precision, p_recall)
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
    return (metrics_list, class_report, plots)


def rf(input_features, input_labels):
    X = np.array(input_features)
    y = input_labels['labels']  
    print(f'Sanity Check\nNumber of embeddings loaded: {len(X)}\nNumber of matching labels: {len(y)}\n')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob_test)
    metrics = [accuracy_score(y_test, y_pred), *[np.mean(x) for x in precision_recall_fscore_support(y_test, y_pred, zero_division=0, average='macro')]]
    return(metrics, classification_report(y_test, y_pred, zero_division=0), (precision, recall))

def load_embeddings(file_path):
    """
    Load document embeddings from a .npy file.
    """
    embeddings_array = np.load(file_path)
    print(f"Embeddings loaded from {file_path}")
    return embeddings_array

# Load the extracted feature embeddings from glove
# loaded_embeddings = load_embeddings('document_embeddings.npy')
# loaded_embeddings_tfidf = load_embeddings('document_embeddings_tfidf.npy')
