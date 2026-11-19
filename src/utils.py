import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

class ConfigLoader:
    def __init__(self):
        self.config = {}
    
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        return self.config
    
    def get_model_config(self, model_name):
        return self.config.get('models', {}).get(model_name, {})
    
    def get_preprocessing_config(self):
        return self.config.get('preprocessing', {})

class ResultsLogger:
    def __init__(self, log_dir='models/training_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_training_history(self, model_name, history, dataset_name):
        log_file = os.path.join(self.log_dir, f"{model_name}_{dataset_name}_{self.timestamp}.json")
        
        log_data = {
            'model': model_name,
            'dataset': dataset_name,
            'timestamp': self.timestamp,
            'history': history
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def log_metrics(self, model_name, metrics, dataset_name):
        log_file = os.path.join(self.log_dir, f"metrics_{model_name}_{dataset_name}_{self.timestamp}.json")
        
        log_data = {
            'model': model_name,
            'dataset': dataset_name,
            'timestamp': self.timestamp,
            'metrics': metrics
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

def generate_synthetic_dataset(dataset_type, num_samples=1000):
    np.random.seed(42)
    
    mental_health_keywords = [
        'depression', 'anxiety', 'therapy', 'mental', 'health', 'stress',
        'sad', 'happy', 'medication', 'counseling', 'psychologist',
        'bipolar', 'disorder', 'panic', 'attack', 'suicidal',
        'recovery', 'wellness', 'mindfulness', 'meditation'
    ]
    
    toxic_keywords = [
        'hate', 'stupid', 'idiot', 'worthless', 'kill', 'die',
        'useless', 'terrible', 'awful', 'disgusting', 'horrible'
    ]
    
    neutral_keywords = [
        'weather', 'food', 'movie', 'music', 'sports', 'travel',
        'work', 'school', 'family', 'friends', 'shopping'
    ]
    
    if dataset_type == 'dataset1':
        texts = []
        labels = []
        
        for i in range(num_samples):
            if np.random.random() > 0.5:
                words = np.random.choice(mental_health_keywords, size=np.random.randint(5, 15))
                text = ' '.join(words) + ' ' + ' '.join(['word' + str(i) for i in range(np.random.randint(3, 8))])
                labels.append(1)
            else:
                words = np.random.choice(neutral_keywords, size=np.random.randint(5, 15))
                text = ' '.join(words) + ' ' + ' '.join(['term' + str(i) for i in range(np.random.randint(3, 8))])
                labels.append(0)
            texts.append(text)
        
        return pd.DataFrame({'text': texts, 'label': labels})
    
    elif dataset_type == 'dataset2':
        texts = []
        labels = []
        
        for i in range(num_samples):
            if np.random.random() > 0.7:
                words = np.random.choice(toxic_keywords, size=np.random.randint(3, 10))
                text = ' '.join(words) + ' ' + ' '.join(['text' + str(i) for i in range(np.random.randint(5, 12))])
                labels.append(1)
            else:
                words = np.random.choice(neutral_keywords, size=np.random.randint(5, 15))
                text = ' '.join(words) + ' ' + ' '.join(['content' + str(i) for i in range(np.random.randint(3, 8))])
                labels.append(0)
            texts.append(text)
        
        return pd.DataFrame({'text': texts, 'label': labels})
    
    elif dataset_type == 'dataset3':
        texts = []
        labels = []
        conditions = ['stress', 'depression', 'bipolar', 'personality_disorder', 'anxiety']
        
        for i in range(num_samples):
            condition = np.random.choice(conditions)
            base_words = []
            
            if condition == 'stress':
                base_words = ['stress', 'pressure', 'workload', 'deadline', 'overwhelmed']
            elif condition == 'depression':
                base_words = ['depression', 'sad', 'hopeless', 'empty', 'tired']
            elif condition == 'bipolar':
                base_words = ['bipolar', 'mood', 'swings', 'mania', 'depressive']
            elif condition == 'personality_disorder':
                base_words = ['personality', 'disorder', 'borderline', 'unstable', 'relationships']
            elif condition == 'anxiety':
                base_words = ['anxiety', 'worry', 'panic', 'fear', 'nervous']
            
            additional_words = np.random.choice(mental_health_keywords, size=np.random.randint(5, 10))
            all_words = base_words + additional_words.tolist()
            np.random.shuffle(all_words)
            
            text = ' '.join(all_words) + ' ' + ' '.join(['phrase' + str(i) for i in range(np.random.randint(3, 8))])
            texts.append(text)
            labels.append(conditions.index(condition))
        
        return pd.DataFrame({'text': texts, 'label': labels})
    
    else:
        raise ValueError("dataset_type must be 'dataset1', 'dataset2', or 'dataset3'")