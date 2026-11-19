import tensorflow as tf
from transformers import (
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer, 
    RobertaTokenizer,
    XLNetTokenizer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

class TransformerTrainer:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.tokenizers = {}
        self.histories = {}
    
    def get_model_and_tokenizer(self, model_name):
        model_configs = {
            'bert': ('bert-base-uncased', BertTokenizer),
            'roberta': ('roberta-base', RobertaTokenizer),
            'distilbert': ('distilbert-base-uncased', AutoTokenizer),
            'xlnet': ('xlnet-base-cased', XLNetTokenizer)
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        model_path, tokenizer_class = model_configs[model_name]
        tokenizer = tokenizer_class.from_pretrained(model_path)
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=2
        )
        
        return model, tokenizer
    
    def prepare_dataset(self, texts, labels, tokenizer):
        encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=self.config['models'][list(self.config['models'].keys())[0]]['max_length'],
            return_tensors='tf'
        )
        
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            labels
        ))
        
        return dataset
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        model, tokenizer = self.get_model_and_tokenizer(model_name)
        
        train_dataset = self.prepare_dataset(X_train, y_train, tokenizer)
        val_dataset = self.prepare_dataset(X_val, y_val, tokenizer)
        
        batch_size = self.config['models'][model_name]['batch_size']
        train_dataset = train_dataset.batch(batch_size).shuffle(1000)
        val_dataset = val_dataset.batch(batch_size)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['models'][model_name]['learning_rate']
        )
        
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config['models'][model_name]['epochs'],
            verbose=1
        )
        
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        self.histories[model_name] = history.history
        
        return model, history.history
    
    def evaluate_model(self, model_name, X_test, y_test):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        test_dataset = self.prepare_dataset(X_test, y_test, tokenizer)
        test_dataset = test_dataset.batch(self.config['models'][model_name]['batch_size'])
        
        predictions = model.predict(test_dataset)
        predicted_labels = np.argmax(predictions.logits, axis=1)
        
        accuracy = accuracy_score(y_test, predicted_labels)
        precision = precision_score(y_test, predicted_labels, average='weighted')
        recall = recall_score(y_test, predicted_labels, average='weighted')
        f1 = f1_score(y_test, predicted_labels, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predicted_labels
        }
    
    def save_model(self, model_name, path):
        if model_name in self.models:
            self.models[model_name].save_pretrained(path)
            self.tokenizers[model_name].save_pretrained(path)