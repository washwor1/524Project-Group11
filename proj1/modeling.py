import numpy as np 
import pandas as pd
from contextlib import redirect_stdout


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TextVectorization;

import os

def gaussian(input_features, input_labels):
    X = np.array(input_features)
    y = input_labels['labels']  
    print(f'Sanity Check\nNumber of embeddings loaded: {len(X)}\nNumber of matching labels: {len(y)}\n')
    model = GaussianNB()
    Xtrain,Xtest,ytrain,ytest = train_test_split(X, y, train_size= 0.8, random_state=42)
    model.fit(Xtrain, ytrain)
    NB_pre = model.predict(Xtest)
    return(accuracy_score(NB_pre,ytest), classification_report(ytest, NB_pre))

def svm(input_features, input_labels):
    X = np.array(input_features)
    y = input_labels['labels']  
    print(f'Sanity Check\nNumber of embeddings loaded: {len(X)}\nNumber of matching labels: {len(y)}\n')
    model_SVC = SVC(kernel = "sigmoid") # try different kernel
    Xtrain,Xtest,ytrain,ytest = train_test_split(X, y, train_size= 0.8, random_state=42)
    model_SVC.fit(Xtrain, ytrain)
    SVC_pre = model_SVC.predict(Xtest)
    return(accuracy_score(SVC_pre,ytest), classification_report(ytest, SVC_pre))

def lstm(input_features, input_labels):
    # y = input_labels['labels']  
    # print(f'Sanity Check\nNumber of embeddings loaded: {len(X)}\nNumber of matching labels: {len(y)}\n')
    # Xtrain,Xtest,ytrain,ytest = train_test_split(X, y, train_size= 0.8)
    Xtrain_lstm = input_features
    Xtrain_lstm = np.array(input_features).reshape((Xtrain_lstm.shape[0], 1, Xtrain_lstm.shape[1]))
    lstm = Sequential()
    lstm.add(LSTM(units = 128, input_shape=(Xtrain_lstm.shape[1], Xtrain_lstm.shape[2])))
    lstm.add(Dense(1, activation='sigmoid'))
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm.summary()
    Xtrain_lstm,Xtest_lstm,ytrain_lstm,ytest_lstm = train_test_split(Xtrain_lstm, input_labels, train_size= 0.8, random_state=42)
    print(Xtrain_lstm.shape)
    print(ytrain_lstm.shape)
    lstm.fit(Xtrain_lstm, ytrain_lstm, epochs=10, batch_size=32, validation_split=0.2)
    loss, accuracy = lstm.evaluate(Xtest_lstm, ytest_lstm)
    return (loss, accuracy)

def rf(input_features, input_labels):
    X = np.array(input_features)
    y = input_labels['labels']  
    print(f'Sanity Check\nNumber of embeddings loaded: {len(X)}\nNumber of matching labels: {len(y)}\n')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return(accuracy_score(y_pred,y_test), classification_report(y_test, y_pred))


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
    # tfidf_features_path = os.path.join(current_dir, 'data', 'all_features.csv')
    # tfidf_features = pd.read_csv(tfidf_features_path)

    labels_path = os.path.join(current_dir, 'data', 'all_labels.csv')

    # Load the data
    Labels = pd.read_csv(labels_path)

    # Load the extracted feature embeddings from glove
    loaded_embeddings = load_embeddings('document_embeddings.npy')


    functions = [rf, gaussian, svm, lstm]
    os.makedirs("modelTesting", exist_ok=True)
    print("Testing Models with glove embeddings")
    for function in functions:
        with open(f'modelTesting/{function.__name__}-glove.txt', 'w') as f:
            with redirect_stdout(f):
                print(f"Experiment Results: ")
                try:
                    accuracy, class_report = function(loaded_embeddings, Labels)
                except Exception as e:
                    print(e)
                print(accuracy)
                print(class_report)
    
    # print("Testing Models with tfidf features")
    # for function in functions:
    #     with open(f'modelTesting/{function.__name__}-tfidf.txt', 'w') as f:
    #         with redirect_stdout(f):
    #             print(f"Experiment Results: ")
    #             try: 
    #                 accuracy, class_report = function(tfidf_features, Labels)
    #             except Exception as e:
    #                 print(e)
    #             print(accuracy)
    #             print(class_report)

if __name__ == "__main__":
    model_data()
