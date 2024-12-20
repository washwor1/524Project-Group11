'''
    Contains classical ML models for the experiment
    (adapted from Project 1)
'''
import os
import time
import sys

import numpy as np 
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.autograd import Variable 
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.functions import array_to_vector
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, StringType

from proj_logging import logger

def spark_models(file_path, feature_type):
    '''
    Runs the PySpark versions of Naive Bayes and Random Forest
    which are much more performant than the scikit-learn ones
    '''
    logger.info("Running PySpark models (Naive Bayes, RF)")
    conf = SparkConf().setMaster("local[*]").setAppName("SparkTFIDF") \
            .set('spark.local.dir', '/media/volume/team11data/tmp') \
            .set('spark.driver.memory', '50G') \
            .set('spark.driver.maxResultSize', '25G') \
            .set('spark.executor.memory', '10G')
    sc = SparkContext(conf = conf)
    sc.setLogLevel("ERROR")
    spark = SparkSession(sc)
    dataset = spark.read.parquet(file_path)    
    dataset = dataset.select(col("author_id").cast(StringType()).alias('author_id'), array_to_vector("features").alias("features"), "is_train")
    label_indexer = StringIndexer(inputCol="author_id", outputCol = "label")
    model = label_indexer.fit(dataset)
    dataset = model.transform(dataset)
    data_dir = f"{file_path[:file_path.rindex('/')]}/{feature_type}"
    dataset.select("author_id", "label").distinct().write.csv(f"{data_dir}/spark_author_mapping.csv", mode='overwrite')
    # call model
    funcs = [gaussian, rf]
    results = []
    try:
        for f in funcs:
            sub_dir = f"{data_dir}/{f.__name__}"
            os.makedirs(sub_dir, exist_ok=True)
            start_time = time.time()
            preds_and_labels = f(dataset)
            end_time = time.time()
            metrics = MulticlassMetrics(preds_and_labels.rdd)
            cm = metrics.confusionMatrix().toArray()
            logger.debug(cm)
            np.savetxt(f"{sub_dir}/confusion_matrix.txt", cm)
            preds_and_labels.write.parquet(f"{sub_dir}/raw_preds_and_labels.parquet", mode='overwrite', partitionBy='label')
            arr = [metrics.accuracy, metrics.weightedFMeasure(), metrics.weightedPrecision, metrics.weightedRecall]
            logger.debug(arr)
            results.append([f.__name__, feature_type, 'test', (end_time - start_time), *arr])
    except Exception as e:
        spark.stop()
        logger.error(e)
        raise e

    # end spark
    logger.debug(results)
    spark.stop()
    return results

def gaussian(df):
    logger.info("Testing Naive Bayes Classifier")
    train_df = df.filter(df['is_train'] == False)  # noqa: E712
    clf = NaiveBayes(featuresCol='features', labelCol='label')
    model = clf.fit(train_df)
    test_df = df.filter(df['is_train'] == True)# noqa: E712
    results = model.transform(test_df)
    preds_and_labels = results.select(['prediction','label']).withColumn('label', col('label').cast(FloatType())).orderBy('prediction')
    preds_and_labels = preds_and_labels.select(['prediction','label'])
    return preds_and_labels

def rf(df):
    logger.info("Testing Random Forest Classifier")
    train_df = df.filter(df['is_train'] == False)# noqa: E712
    clf = RandomForestClassifier(featuresCol='features', labelCol='label')
    model = clf.fit(train_df)
    test_df = df.filter(df['is_train'] == True)# noqa: E712
    results = model.transform(test_df)
    preds_and_labels = results.select(['prediction','label']).withColumn('label', col('label').cast(FloatType())).orderBy('prediction')
    preds_and_labels = preds_and_labels.select(['prediction','label'])
    return preds_and_labels

def svm(X_train, X_test, y_train, y_test, data_dir):
    '''
    Runs Support Vector Classifier (a type of Support Vector Machine)
    '''
    logger.debug("Starting SVM Testing")
    clf = SVC(kernel = "sigmoid", probability=True) # try different kernel
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # y_prob_test = clf.predict_proba(X_test)[:,1]
    # precision, recall, thresholds = precision_recall_curve(y_test, y_prob_test)
    metrics = [accuracy_score(y_test, y_pred), *[np.mean(x) for x in precision_recall_fscore_support(y_test, y_pred, zero_division=0, average='macro') if x is not None]]
    logger.debug("Finished SVM Testing")
    logger.debug([metrics[0], metrics[3], metrics[1], metrics[2]])
    
    # Confusion Matrix
    preds_and_labels = pd.DataFrame([], columns=['pred', 'label'])
    preds_and_labels['pred'] = y_pred
    preds_and_labels['label'] = y_test
    tbl = pa.Table.from_pandas(preds_and_labels)
    pq.write_to_dataset(tbl, f"{data_dir}/svm_preds_and_labels.parquet", partition_cols=['label'])
    
    return [metrics[0], metrics[3], metrics[1], metrics[2]]

class LSTM_Torch(nn.Module):
    '''
    PyTorch version of LSTM 

    Based on:
    - https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    - https://cnvrg.io/pytorch-lstm/
    '''
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_Torch, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, num_classes) #fully connected 1
        # self.fc = nn.Linear(128, 1) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        tag_space = self.fc_1(output.view(len(x), -1))
        scores = log_softmax(tag_space, dim=1)
        return scores

def lstm(t_train, t_test, y_train, y_test, data_dir):
    num_epochs = 10
    logger.info("LSTM - Beginning run")
    t_tensor = torch.Tensor(t_train)
    y_tensor = torch.Tensor(y_train)

    t_final = torch.reshape(t_tensor, (t_tensor.shape[0],1,t_tensor.shape[1]))
    dataset = TensorDataset(t_final, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    t_test_tensor = torch.Tensor(t_test)
    y_test_tensor = torch.Tensor(y_test)
    t_test_final = torch.reshape(t_test_tensor, (t_test_tensor.shape[0],1,t_test_tensor.shape[1]))
    test_dataset = TensorDataset(t_test_final, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    model = LSTM_Torch(4, t_tensor.shape[1], 2, 1, 4)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.NLLLoss()
    for epoch in range(0, num_epochs):
        total_loss = 0
        for data, labels in dataloader:
            labels = labels.type(torch.LongTensor).to(device)
            data = data.to(device)
        
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_loss:.4f}")

    all_preds = []
    all_labels =[] 
    with torch.no_grad():
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.type(torch.LongTensor).to(device)
    
            preds = model(data)
            test_preds = torch.argmax(preds, axis=1)
            all_preds.extend(test_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    logger.info(all_preds[:20])
    logger.info(all_labels[:20])
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    preds_and_labels = pd.DataFrame([], columns=['pred', 'label'])
    preds_and_labels['pred'] = all_preds
    preds_and_labels['label'] = all_labels
    tbl = pa.Table.from_pandas(preds_and_labels)
    pq.write_to_dataset(tbl, f"{data_dir}/lstm_preds_and_labels.parquet", partition_cols=['label'])
    
    logger.debug([accuracy, f1, precision, recall])
    return [accuracy, f1, precision, recall]



if __name__ == "__main__":
    CONFIG_NAME = "all"
    if len(sys.argv) == 2:
        CONFIG_NAME = str(sys.argv[1])
    
    logger.info(f"Creating features (config = '{CONFIG_NAME}')")
    # start = time.time()
    spark_models(f"data/{CONFIG_NAME}/document_embeddings.parquet")