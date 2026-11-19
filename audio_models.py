import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import yaml
import librosa

class AudioModelTrainer:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.histories = {}
    
    def build_cnn_audio_model(self, input_shape, num_classes):
        """Build CNN model for audio classification using MFCC features"""
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def build_lstm_audio_model(self, input_shape, num_classes):
        """Build LSTM model for audio classification"""
        model = tf.keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def extract_audio_features(self, audio_path):
        """Extract MFCC features from audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.config['audio_models']['cnn_audio']['sample_rate'])
            
            # Ensure fixed length
            target_length = self.config['audio_models']['cnn_audio']['duration'] * sr
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, max(0, target_length - len(audio))))
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=self.config['audio_models']['cnn_audio']['n_mfcc']
            )
            
            # Normalize
            mfcc = (mfcc - mfcc.mean()) / mfcc.std()
            
            return mfcc
            
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
            return None
    
    def train_model(self, model_type, X_train, y_train, X_val, y_val):
        """Train audio classification model"""
        if model_type not in ['cnn_audio', 'lstm_audio']:
            raise ValueError("Model type must be 'cnn_audio' or 'lstm_audio'")
        
        num_classes = len(np.unique(y_train))
        
        if model_type == 'cnn_audio':
            # Reshape for CNN (add channel dimension)
            X_train_reshaped = X_train[..., np.newaxis]
            X_val_reshaped = X_val[..., np.newaxis]
            model = self.build_cnn_audio_model(X_train_reshaped.shape[1:], num_classes)
        else:
            model = self.build_lstm_audio_model(X_train.shape[1:], num_classes)
            X_train_reshaped = X_train
            X_val_reshaped = X_val
        
        # Compile model
        if num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['audio_models'][model_type]['learning_rate']),
            loss=loss,
            metrics=metrics
        )
        
        # Train model
        history = model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=self.config['audio_models'][model_type]['epochs'],
            batch_size=self.config['audio_models'][model_type]['batch_size'],
            verbose=1
        )
        
        self.models[model_type] = model
        self.histories[model_type] = history.history
        
        return model, history.history
    
    def evaluate_model(self, model_type, X_test, y_test):
        """Evaluate audio classification model"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        model = self.models[model_type]
        
        if model_type == 'cnn_audio':
            X_test_reshaped = X_test[..., np.newaxis]
        else:
            X_test_reshaped = X_test
        
        predictions = model.predict(X_test_reshaped)
        
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
        """Save trained model"""
        if model_type in self.models:
            self.models[model_type].save(path)