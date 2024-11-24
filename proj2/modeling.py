import logging
import pyarrow as pa
import os
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import time
import random 
import sys
import json

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from classical_models import rf, svm, lstm, gaussian, spark_models
from feature_engineering import extract_features
from dataset_handling import book_train_test_split, load_dataset
from p_logging import logger

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
            logger.info("tttt")


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
        
        train_df = df[df['is_train']].reset_index(drop=True)
        test_df = df[~df['is_train']].reset_index(drop=True)

        train_dataset = CustomDataset(train_df.text, train_df.author_id, tokenizer, max_len)
        test_dataset = CustomDataset(test_df.text, test_df.author_id, tokenizer, max_len)
        
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=16)

    
    def fit(self):
        # fit transformer
        self.logger.info(f"Fitting Transformer")
        self.start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
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

        # Confusion Metrics
        preds_and_labels = pd.DataFrame([], columns=['pred', 'label'])
        preds_and_labels['pred'] = all_preds
        preds_and_labels['label'] = all_labels
        tbl = pa.Table.from_pandas(preds_and_labels)
        pq.write_to_dataset(tbl, f"data/{self.config_name}/transformer_preds_and_labels.parquet", partition_cols=['label'])
        
        self.logger.debug([['transformer', 'embeddings', 'test', duration, accuracy, f1, precision, recall]])
        return [['transformer', 'embeddings', 'test', duration, accuracy, f1, precision, recall]]

class ClassicalModels(Model):
    def create_features(self, df: pd.DataFrame, config_name: str):
        self.data_dir = f"data/{config_name}"
        # raw_tfidf, raw_embeddings = extract_features(f"{self.data_dir}/dataset.parquet", config_name=config_name)

        # self.tfidf = pd.read_parquet(f"{self.data_dir}/tfidf_features.parquet")
        # self.embeddings = pd.read_parquet(f"{self.data_dir}/document_embeddings.parquet")
        # self.embeddings = pd.DataFrame(raw_embeddings['features'].apply(lambda x: x['values']).tolist())
        
        # self.tfidf = self.tfidf.fillna(0)

    def fit(self):
        '''
        To avoid reusing code, this function does nothing, as
        the per-model functions already train and then test
        '''
        return None
    
    def predict(self):
        # run all models and return metrics
        # functions = [rf, gaussian, svm, lstm]
        functions = [lstm, svm]
        metrics_arr = []
        for feature_type in ['glove', 'tfidf']:
            self.logger.info(f"Processing {feature_type} features")
            path = "tfidf_features.parquet" if feature_type == "tfidf" else "document_embeddings.parquet"
            sub_dir = f"{self.data_dir}/{feature_type}"
            os.makedirs(sub_dir, exist_ok=True)
            metrics_arr += spark_models(f"{self.data_dir}/{path}", feature_type)

            self.tfidf = pd.read_parquet(f"{self.data_dir}/tfidf_features.parquet")
            self.embeddings = pd.read_parquet(f"{self.data_dir}/document_embeddings.parquet")

            id_to_label ={int(x): i for i, x in enumerate(self.tfidf.author_id.unique())}
            with open(f"{self.data_dir}/lstm_svm_id_to_label.json", 'w') as fp:
                json.dump(id_to_label, fp)
            features = self.tfidf if feature_type == "tfidf" else self.embeddings
            features['label'] = features.author_id.apply(lambda x: id_to_label[x])            
            X_train = np.array(features[features.is_train].features.tolist())
            X_test = np.array(features[~features.is_train].features.tolist())
            y_train = features[features.is_train].label.to_numpy()
            y_test = features[~features.is_train].label.to_numpy()

            for function in functions:
                try:
                    start_time = time.time()
                    self.logger.info(f"Beginning testing of {function.__name__} with {feature_type} features")
                    metrics = function(X_train, X_test, y_train, y_test, sub_dir)
                    # self.logger.info(classification_report)
                    self.logger.info(f"Finished testing of {function.__name__} with {feature_type} features (took {time.time() - start_time} seconds)")
                    metrics_arr.append([function.__name__, feature_type, 'test', (time.time() - start_time), *metrics])
                except Exception as e:
                    raise e  
            self.logger.info(f"Finished processing {feature_type} features")
        return metrics_arr