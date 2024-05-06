from datasets import load_dataset
from transformers import AutoTokenizer
import re

class DataProcessing():
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
        self.raw_train, self.raw_test = self.load_csv_files()
        self.train_dataset, self.test_dataset = self.prepare_datasets()

    def load_csv_files(self):
        # Load the training and testing files in HF dataset format
        raw_train = load_dataset("csv", data_files={'train': [self.train_file]})
        raw_test = load_dataset("csv", data_files={'test': [self.test_file]})
        return raw_train, raw_test

    def preprocess_function(self, data):
        # Preprocessing function
        texts = (data["text"],)
        processed_texts = []
        for text in texts[0]:
            text = re.sub(r'http[s]?://\S+', '', text)
            text = re.sub(r' www\S+', '', text)
            text = re.sub(r'@\S+', '', text)
            text = re.sub(r'[^\w\s]|[\d]', ' ', text)
            text = re.sub(r'\s\s+', ' ', text)
            text = text.strip().lower().encode('ascii', 'ignore').decode()
            processed_texts.append(text)
        processed = self.tokenizer(processed_texts, padding="max_length", max_length=128, truncation=True)
        processed["labels"] = data["sentiment"]
        return processed

    def prepare_datasets(self):
        # Map preprocessing function and remove original columns
        train_dataset = self.raw_train.map(self.preprocess_function, batched=True, remove_columns=self.raw_train["train"].column_names)
        test_dataset = self.raw_test.map(self.preprocess_function, batched=True, remove_columns=self.raw_test["test"].column_names)
        return train_dataset, test_dataset

    def get_data_splits(self):
        # Select subsets for training and validation
        train = self.train_dataset['train'].select(range(200))
        valid = self.test_dataset['test'].select(range(100))
        infer = self.test_dataset['test']
        return train, valid, infer