import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import yaml

class ImageModelTrainer:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.histories = {}
    
    def build_efficientnet_model(self, num_classes):
        """Build EfficientNet model for image classification"""
        base_model = EfficientNetB0(
            include_top=False,
            weights=self.config['image_models']['efficientnet']['weights'],
            input_shape=self.config['image_models']['efficientnet']['input_size']
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def build_resnet_model(self, num_classes):
        """Build ResNet50 model for image classification"""
        base_model = ResNet50(
            include_top=False,
            weights=self.config['image_models']['resnet50']['weights'],
            input_shape=self.config['image_models']['resnet50']['input_size']
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def build_cnn_model(self, input_shape, num_classes):
        """Build custom CNN model for image classification"""
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def train_model(self, model_type, X_train, y_train, X_val, y_val):
        """Train image classification model"""
        if model_type not in ['efficientnet', 'resnet50', 'cnn']:
            raise ValueError("Model type must be 'efficientnet', 'resnet50', or 'cnn'")
        
        num_classes = len(np.unique(y_train))
        
        if model_type == 'efficientnet':
            model = self.build_efficientnet_model(num_classes)
        elif model_type == 'resnet50':
            model = self.build_resnet_model(num_classes)
        else:
            model = self.build_cnn_model(X_train.shape[1:], num_classes)
        
        # Compile model
        if num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['image_models'][model_type]['learning_rate']),
            loss=loss,
            metrics=metrics
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['image_models'][model_type]['epochs'],
            batch_size=self.config['image_models'][model_type]['batch_size'],
            verbose=1
        )
        
        self.models[model_type] = model
        self.histories[model_type] = history.history
        
        return model, history.history
    
    def evaluate_model(self, model_type, X_test, y_test):
        """Evaluate image classification model"""
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