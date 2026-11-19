import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import yaml
import cv2

class VideoModelTrainer:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.histories = {}
    
    def build_3d_cnn_model(self, input_shape, num_classes):
        """Build 3D CNN model for video classification"""
        model = tf.keras.Sequential([
            layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling3D((1, 2, 2)),
            layers.Conv3D(64, (3, 3, 3), activation='relu'),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Conv3D(128, (3, 3, 3), activation='relu'),
            layers.GlobalAveragePooling3D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def build_cnn_lstm_model(self, input_shape, num_classes):
        """Build CNN-LSTM model for video classification"""
        # CNN feature extractor
        cnn_base = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape[1:]  # Exclude temporal dimension
        )
        cnn_base.trainable = False
        
        model = tf.keras.Sequential([
            layers.TimeDistributed(cnn_base, input_shape=input_shape),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def extract_video_frames(self, video_path, max_frames=30):
        """Extract frames from video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
            
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in frame_indices:
                    # Resize and normalize frame
                    frame = cv2.resize(frame, self.config['video_models']['3d_cnn']['frame_size'][:2])
                    frame = frame / 255.0
                    frames.append(frame)
            
            cap.release()
            
            # Pad if necessary
            while len(frames) < max_frames:
                frames.append(np.zeros(self.config['video_models']['3d_cnn']['frame_size']))
            
            return np.array(frames)
            
        except Exception as e:
            print(f"Error processing video file {video_path}: {e}")
            return None
    
    def train_model(self, model_type, X_train, y_train, X_val, y_val):
        """Train video classification model"""
        if model_type not in ['3d_cnn', 'cnn_lstm']:
            raise ValueError("Model type must be '3d_cnn' or 'cnn_lstm'")
        
        num_classes = len(np.unique(y_train))
        
        if model_type == '3d_cnn':
            model = self.build_3d_cnn_model(X_train.shape[1:], num_classes)
        else:
            model = self.build_cnn_lstm_model(X_train.shape[1:], num_classes)
        
        # Compile model
        if num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['video_models'][model_type]['learning_rate']),
            loss=loss,
            metrics=metrics
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['video_models'][model_type]['epochs'],
            batch_size=self.config['video_models'][model_type]['batch_size'],
            verbose=1
        )
        
        self.models[model_type] = model
        self.histories[model_type] = history.history
        
        return model, history.history
    
    def evaluate_model(self, model_type, X_test, y_test):
        """Evaluate video classification model"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        model = self.models[model_type]
        predictions = model.predict(X_test)
        
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