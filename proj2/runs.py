import logging
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import time
import random 
import sys

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from modeling import rf, svm, lstm, gaussian
from feature_engineering import extract_features
from dataset_handling import book_train_test_split, load_dataset
from p_logging import logger

LOGGER_NAME = "proj2_logger"

class Model:
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
        new_cats = {x: i for i, x in enumerate(unique)}
        df.author_id = df.author_id.cat.rename_categories(new_cats)
        self.num_labels = len(unique)
        self.logger.info(df.author_id.unique())
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
        # Evaluation
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
        
        # Metric
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return [['transformer', 'embeddings', 'test', duration, accuracy, f1, precision, recall]]

class ClassicalModels(Model):
    def create_features(self, df: pd.DataFrame, config_name: str):
        self.tfidf, self.embeddings = extract_features(f"data/{config_name}/dataset.parquet")
        self.train_mapping = df['is_train']
        
        self.labels = df['author_id']

    def fit(self):
        '''
        To avoid reusing code, this function does nothing, as
        the per-model functions already train and then test
        '''
        return None
    
    def predict(self):
        # run all models and return metrics
        functions = [rf, gaussian, svm, lstm]
        metrics_arr = []
        for feature_type in ['glove', 'tfidf']:
            self.logger.debug(f"Processing {feature_type} features")
            features = self.tfidf if feature_type == "tfidf" else self.embeddings
            X_train = features[self.train_mapping]
            X_test = features[~self.train_mapping]
            y_train = self.labels[self.train_mapping]
            y_test = self.labels[~self.train_mapping]
            for function in functions:
                try:
                    start_time = time.time()
                    self.logger.debug(f"Beginning testing of {function.__name__} with {feature_type} features")
                    metrics, classification_report, pr = function(X_train, X_test, y_train, y_test)
                    self.logger.debug(classification_report)
                    self.logger.debug(f"Finished testing of {function.__name__} with {feature_type} features (took {time.time() - start_time} seconds)")
                    metrics_arr.append([function.__name__, feature_type, 'test', *metrics])
                except Exception as e:
                    print(e)  
            self.logger.debug(f"Finished processing {feature_type} features")
        return metrics_arr

def experiment(config_name):
    # load dataset
    # run train-test-split on df (will produce label column)
    df = load_dataset(config_name=config_name)
    if config_name == "primary_authors":
        df = book_train_test_split(df)
    elif config_name == "all":
        df = book_train_test_split(df, margin_of_error=0.05)
    models = [TransformerModel()]
    metrics = []
    for model in models:
        model.create_features(df, config_name)
        model.fit()
        metrics += model.predict()

    metrics_df = pd.DataFrame(metrics, columns=['model_name', 'data_type', 'phase', 'duration', 'accuracy', 'f1-score', 'precision', 'recall'])
    metrics_df.to_csv(f"data/{config_name}/metrics.csv", index=False)
if __name__ == "__main__":
    CONFIG_NAME = "all"
    if len(sys.argv) == 2:
        CONFIG_NAME = str(sys.argv[1])
    
    logger.info(f"Running models (config = '{CONFIG_NAME}')")
    start = time.time()
    experiment(CONFIG_NAME)
    logger.info(f"Finished running experiment (took {(time.time() - start):.4f} seconds)")
