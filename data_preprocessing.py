import pandas as pd
import numpy as np
import re
import nltk
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import AutoTokenizer
import yaml

class DataPreprocessor:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.nlp = spacy.load('en_core_web_sm')
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_and_lemmatize(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in self.stop_words and not token.is_space]
        return ' '.join(tokens)
    
    def preprocess_dataset(self, df, text_column, label_column):
        df_clean = df.copy()
        df_clean['cleaned_text'] = df_clean[text_column].apply(self.clean_text)
        df_clean['processed_text'] = df_clean['cleaned_text'].apply(self.tokenize_and_lemmatize)
        
        if label_column in df_clean.columns:
            le = LabelEncoder()
            df_clean['encoded_labels'] = le.fit_transform(df_clean[label_column])
            label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        else:
            df_clean['encoded_labels'] = 0
            label_mapping = {}
            
        return df_clean, label_mapping
    
    def prepare_train_test_split(self, df, text_column='processed_text', label_column='encoded_labels'):
        X_train, X_test, y_train, y_test = train_test_split(
            df[text_column],
            df[label_column],
            test_size=self.config['preprocessing']['test_size'],
            random_state=self.config['preprocessing']['random_state'],
            stratify=df[label_column] if len(df[label_column].unique()) > 1 else None
        )
        return X_train, X_test, y_train, y_test
    
    def prepare_tfidf_features(self, X_train, X_test):
        vectorizer = TfidfVectorizer(
            max_features=self.config['preprocessing']['max_features']
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf, vectorizer
    
    def prepare_transformer_features(self, texts, model_name='bert-base-uncased'):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=self.config['preprocessing']['sequence_length'],
            return_tensors='tf'
        )
        return encodings