'''
    Contains classical ML models for the experiment
    (adapted from Project 1)
'''
import numpy as np 
import pandas as pd
from contextlib import redirect_stdout
from p_logging import logger

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support ,precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import tensorflow as tf
import os

def gaussian(X_train, X_test, y_train, y_test):
    '''
    Runs Gaussian Naive Bayes model 
    '''
    logger.debug("Starting Naive Bayes testing")
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    logger.debug("Fit model")
    y_pred = clf.predict(X_test)
    logger.debug("Predicted test data")
    metrics = [accuracy_score(y_test, y_pred), *[np.mean(x) for x in precision_recall_fscore_support(y_test, y_pred, zero_division=0, average='macro')]]
    y_prob_test = clf.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob_test)
    logger.debug("Finished Naive Bayes testing")
    return [metrics[0], metrics[3], metrics[1], metrics[2]]
    # return(metrics, classification_report(y_test, y_pred, zero_division=0), (precision, recall))

def svm(X_train, X_test, y_train, y_test):
    '''
    Runs Support Vector Classifier (a type of Support Vector Machine)
    '''
    logger.debug("Starting SVM Testing")
    clf = SVC(kernel = "sigmoid", probability=True) # try different kernel
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob_test)
    metrics = [accuracy_score(y_test, y_pred), *[np.mean(x) for x in precision_recall_fscore_support(y_test, y_pred, zero_division=0, average='macro')]]
    logger.debug("Finished SVM Testing")
    return [metrics[0], metrics[3], metrics[1], metrics[2]]

    # return(metrics, classification_report(y_test, y_pred, zero_division=0), (precision, recall))

def lstm(X_train, X_test, y_train, y_test):
    '''
    Runs an LSTM model
    '''
    logger.debug("Started LSTM Testing")
    # Determine the number of unique classes
    num_classes = len(np.unique(y_test))
    logger.debug("Shaping dataset")

    # Prepare input data
    Xtrain_lstm = np.array(X_train)
    Xtrain_lstm = Xtrain_lstm.reshape((Xtrain_lstm.shape[0], 1, Xtrain_lstm.shape[1]))
    Xtest_lstm = np.array(X_train)
    Xtest_lstm = Xtest_lstm.reshape((Xtest_lstm.shape[0], 1, Xtest_lstm.shape[1]))
    logger.debug("Finished shaping dataset")

    # Build the LSTM model
    lstm = Sequential()
    lstm.add(Input(shape=(Xtrain_lstm.shape[1], Xtrain_lstm.shape[2])))
    lstm.add(LSTM(units=128))

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


    # Train the model
    lstm.fit(
        Xtrain_lstm, y_train,
        verbose=0,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # Make predictions
    y_pred = lstm.predict(Xtest_lstm, batch_size=64, verbose=0)
    print(y_pred)
    plots = None
    # For multiclass classification, select the class with highest probability
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Compute metrics with appropriate averaging method
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred_classes, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred_classes, average='macro', zero_division=0)
    metrics_list = [accuracy, f1, precision, recall, f1]
    # Generate classification report
    class_report = classification_report(y_test, y_pred_classes, zero_division=0)

    # Return metrics and classification report
    return metrics_list


def rf(X_train, X_test, y_train, y_test):
    logger.debug("Started RF Testing")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob_test)
    metrics = [accuracy_score(y_test, y_pred), *[np.mean(x) for x in precision_recall_fscore_support(y_test, y_pred, zero_division=0, average='macro')]]
    return [metrics[0], metrics[3], metrics[1], metrics[2]]
    # return(metrics, classification_report(y_test, y_pred, zero_division=0), (precision, recall))

