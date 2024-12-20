'''
Contains the helper classes that abstract out the work of training and 
testing the various models.

They are split into two classes:
- Transformer Model - Contains code for the BERT transformer
- Classical Models - Contains code for the LSTM, RF, NB, and SVC models
'''

import os
import time
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.functions import array_to_vector
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, StringType

from classical_models import svm, lstm, spark_models
from proj_logging import logger

class Model:
    '''
    Base class to provide common blueprint for both types of models ran in this experiment
    '''
    def __init__(self):
        global logger
        self.logger = logger

    def create_features(self, df: pd.DataFrame, config_name: str):
        raise NotImplementedError("Function was not implemented in subclass")
    def fit(self) -> None:
        raise NotImplementedError("Function was not implemented in subclass")
    def predict(self) -> []:
        '''
        Run the model against the test partition of the dataset.

        Returns metrics: Time, Accuracy, F1-Score, Precision, Recall
        '''
        raise NotImplementedError("Function was not implemented in subclass")

class CustomDataset(Dataset):
    '''
    PyTorch dataset for use by the BERT Transformer in TransformerModel
    '''
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        try:
            text = self.texts[index]
            label = self.labels[index]
            encoding = self.tokenizer(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            logger.error(e)


class TransformerModel(Model):
    def create_features(self, df: pd.DataFrame, config_name: str):
        # split df into pre-created train-test groups
        unique = df.author_id.unique()
        self.config_name = config_name
        new_cats = {x: i for i, x in enumerate(unique)}
        df.author_id = df.author_id.cat.rename_categories(new_cats)
        self.num_labels = len(unique)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_len = 128
        
        train_df = df[~df['is_train']].reset_index(drop=True)
        test_df = df[df['is_train']].reset_index(drop=True)

        train_dataset = CustomDataset(train_df.text, train_df.author_id, tokenizer, max_len)
        test_dataset = CustomDataset(test_df.text, test_df.author_id, tokenizer, max_len)
        
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=16)

    
    def fit(self):
        # fit transformer
        self.logger.info("Fitting Transformer")
        self.start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu") # uncomment this if you are getting pytorch errors 
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels) 
        self.model = self.model.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Training
        epochs = 3
        for epoch in range(epochs):
            self.logger.info(f"Beginning epoch {epoch + 1}")
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
            avg_loss = total_loss / len(self.train_loader)
            self.logger.info(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_loss:.4f}")
        self.logger.info(f"Finished fitting transformer (took {(time.time() - self.start_time):.4f} seconds)")
        return None

    def predict(self) -> []:
        '''
        Evaluate transformer against test dataset
        '''
        self.logger.info("Testing Transformer")
        s = time.time()
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
        
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, axis=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        duration = time.time() - self.start_time
        self.logger.info(f"Finished testing transformer (took {(time.time() - s):.4f} seconds)")
        self.logger.info(f"Overall Transformer Time (took {duration:.4f} seconds)")
        
        # Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        self.logger.debug([['transformer', 'embeddings', 'test', duration, accuracy, f1, precision, recall]])

        # Confusion Matrix
        preds_and_labels = pd.DataFrame([], columns=['pred', 'label'])
        preds_and_labels['pred'] = all_preds
        preds_and_labels['label'] = all_labels
    
        conf = SparkConf().setMaster("local[*]").setAppName("SparkTFIDF") \
            .set('spark.local.dir', '/media/volume/team11data/tmp') \
            .set('spark.driver.memory', '50G') \
            .set('spark.driver.maxResultSize', '25G') \
            .set('spark.executor.memory', '10G')
        sc = SparkContext(conf = conf)
        sc.setLogLevel("ERROR")
        spark = SparkSession(sc)
        try:
            metrics_df = spark.createDataFrame(preds_and_labels)
            metrics_df.write.parquet(f"data/{self.config_name}/transformer_preds_and_labels.parquet", mode="overwrite", partitionBy='label')
        except Exception as e:
            self.logger.error(e)
        finally:
            spark.stop()
        
        return [['transformer', 'embeddings', 'test', duration, accuracy, f1, precision, recall]]

class ClassicalModels(Model):
    def create_features(self, df: pd.DataFrame, config_name: str):
        self.data_dir = f"data/{config_name}"
        # this function used to load the parquet files, but we moved it 
        # to predict() to simplify load times 

    def fit(self):
        '''
        To avoid reusing code, this function does nothing, as
        the per-model functions already train and then test
        '''
        return None
    
    def predict(self):
        # run all models and return metrics
        functions = [lstm, svm]
        metrics_arr = []
        for feature_type in ['glove', 'tfidf']:
            self.logger.info(f"Processing {feature_type} features")
            path = "tfidf_features.parquet" if feature_type == "tfidf" else "document_embeddings.parquet"
            sub_dir = f"{self.data_dir}/{feature_type}"
            os.makedirs(sub_dir, exist_ok=True)
            
            # run PySpark versions of Random Forest and Naive Bayes classifiers
            # there aren't equivalent for the other two so they are run separately
            metrics_arr += spark_models(f"{self.data_dir}/{path}", feature_type)

            # load features for non-PySpark models
            self.tfidf = pd.read_parquet(f"{self.data_dir}/tfidf_features.parquet")
            self.embeddings = pd.read_parquet(f"{self.data_dir}/document_embeddings.parquet")

            # dump id to label mapping in case anything gets confused
            id_to_label ={int(x): i for i, x in enumerate(self.tfidf.author_id.unique())}
            with open(f"{self.data_dir}/lstm_svm_id_to_label.json", 'w') as fp:
                json.dump(id_to_label, fp)
            
            features = self.tfidf if feature_type == "tfidf" else self.embeddings
            features['label'] = features.author_id.apply(lambda x: id_to_label[x])            
            X_train = np.array(features[~features.is_train].features.tolist())
            X_test = np.array(features[features.is_train].features.tolist())
            y_train = features[~features.is_train].label.to_numpy()
            y_test = features[features.is_train].label.to_numpy()

            for function in functions:
                try:
                    start_time = time.time()
                    self.logger.info(f"Beginning testing of {function.__name__} with {feature_type} features")
                    metrics = function(X_train, X_test, y_train, y_test, sub_dir)
                    self.logger.info(f"Finished testing of {function.__name__} with {feature_type} features (took {time.time() - start_time} seconds)")
                    metrics_arr.append([function.__name__, feature_type, 'test', (time.time() - start_time), *metrics])
                except Exception as e:
                    raise e  
            self.logger.info(f"Finished processing {feature_type} features")
        return metrics_arr