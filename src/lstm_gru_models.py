import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import yaml

class RNNTrainer:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.vectorizers = {}
        self.histories = {}
    
    def create_text_vectorizer(self, X_train):
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.config['preprocessing']['max_features'],
            output_sequence_length=self.config['preprocessing']['sequence_length']
        )
        vectorizer.adapt(X_train)
        return vectorizer
    
    def build_lstm_model(self, vocab_size, num_classes):
        model = Sequential([
            Embedding(
                input_dim=vocab_size,
                output_dim=self.config['models']['lstm']['embedding_dim'],
                input_length=self.config['preprocessing']['sequence_length']
            ),
            LSTM(self.config['models']['lstm']['lstm_units'], return_sequences=False),
            Dropout(0.5),
            Dense(self.config['models']['lstm']['dense_units'], activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        return model
    
    def build_gru_model(self, vocab_size, num_classes):
        model = Sequential([
            Embedding(
                input_dim=vocab_size,
                output_dim=self.config['models']['gru']['embedding_dim'],
                input_length=self.config['preprocessing']['sequence_length']
            ),
            GRU(self.config['models']['gru']['gru_units'], return_sequences=False),
            Dropout(0.5),
            Dense(self.config['models']['gru']['dense_units'], activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        return model
    
    def train_model(self, model_type, X_train, y_train, X_val, y_val):
        if model_type not in ['lstm', 'gru']:
            raise ValueError("Model type must be 'lstm' or 'gru'")
        
        vectorizer = self.create_text_vectorizer(X_train)
        self.vectorizers[model_type] = vectorizer
        
        vocab_size = len(vectorizer.get_vocabulary())
        num_classes = len(np.unique(y_train))
        
        if model_type == 'lstm':
            model = self.build_lstm_model(vocab_size, num_classes)
        else:
            model = self.build_gru_model(vocab_size, num_classes)
        
        optimizer = SGD(learning_rate=self.config['models'][model_type]['learning_rate'])
        
        if num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        X_train_vec = vectorizer(X_train)
        X_val_vec = vectorizer(X_val)
        
        history = model.fit(
            X_train_vec, y_train,
            validation_data=(X_val_vec, y_val),
            epochs=self.config['models'][model_type]['epochs'],
            batch_size=self.config['models'][model_type]['batch_size'],
            verbose=1
        )
        
        self.models[model_type] = model
        self.histories[model_type] = history.history
        
        return model, history.history
    
    def evaluate_model(self, model_type, X_test, y_test):
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        model = self.models[model_type]
        vectorizer = self.vectorizers[model_type]
        
        X_test_vec = vectorizer(X_test)
        predictions = model.predict(X_test_vec)
        
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predicted_labels = np.argmax(predictions, axis=1)
        else:
            predicted_labels = (predictions > 0.5).astype(int).flatten()
        
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
    
    def save_model(self, model_type, path):
        if model_type in self.models:
            self.models[model_type].save(path)