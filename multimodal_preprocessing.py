import pandas as pd
import numpy as np
import cv2
import librosa
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import yaml
import os

class MultimodalPreprocessor:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.label_encoder = LabelEncoder()
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        try:
            image = Image.open(image_path)
            image = image.resize(self.config['preprocessing']['image']['resize'])
            image_array = np.array(image)
            
            # Normalize if required
            if self.config['preprocessing']['image']['normalize']:
                if image_array.max() > 1:
                    image_array = image_array / 255.0
            
            return image_array
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio for model input"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.config['preprocessing']['audio']['sample_rate'])
            
            # Ensure fixed length
            target_length = self.config['preprocessing']['audio']['max_duration'] * sr
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, max(0, target_length - len(audio))))
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=self.config['preprocessing']['audio']['n_mfcc']
            )
            
            # Normalize
            mfcc = (mfcc - mfcc.mean()) / mfcc.std()
            
            return mfcc.T  # Time steps first
            
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            return None
    
    def preprocess_video(self, video_path):
        """Preprocess video for model input"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = self.config['preprocessing']['video']['frames_per_sequence']
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
            
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in frame_indices:
                    # Resize and normalize frame
                    frame = cv2.resize(frame, self.config['preprocessing']['video']['frame_size'])
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame / 255.0
                    frames.append(frame)
            
            cap.release()
            
            # Pad if necessary
            while len(frames) < max_frames:
                frames.append(np.zeros(self.config['preprocessing']['video']['frame_size'] + [3]))
            
            return np.array(frames)
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None
    
    def load_multimodal_dataset(self, metadata_path):
        """Load and preprocess multimodal dataset from metadata CSV"""
        metadata = pd.read_csv(metadata_path)
        
        datasets = {
            'image': {'X': [], 'y': []},
            'audio': {'X': [], 'y': []},
            'video': {'X': [], 'y': []},
            'text': {'X': [], 'y': []}
        }
        
        # Encode labels
        if 'label' in metadata.columns:
            labels = self.label_encoder.fit_transform(metadata['label'])
        else:
            labels = np.zeros(len(metadata))
        
        # Process each modality
        for idx, row in metadata.iterrows():
            label = labels[idx]
            
            # Process image if available
            if 'image_path' in row and pd.notna(row['image_path']):
                image_data = self.preprocess_image(row['image_path'])
                if image_data is not None:
                    datasets['image']['X'].append(image_data)
                    datasets['image']['y'].append(label)
            
            # Process audio if available
            if 'audio_path' in row and pd.notna(row['audio_path']):
                audio_data = self.preprocess_audio(row['audio_path'])
                if audio_data is not None:
                    datasets['audio']['X'].append(audio_data)
                    datasets['audio']['y'].append(label)
            
            # Process video if available
            if 'video_path' in row and pd.notna(row['video_path']):
                video_data = self.preprocess_video(row['video_path'])
                if video_data is not None:
                    datasets['video']['X'].append(video_data)
                    datasets['video']['y'].append(label)
            
            # Process text if available
            if 'text' in row and pd.notna(row['text']):
                datasets['text']['X'].append(row['text'])
                datasets['text']['y'].append(label)
        
        # Convert to numpy arrays
        for modality in datasets:
            if len(datasets[modality]['X']) > 0:
                datasets[modality]['X'] = np.array(datasets[modality]['X'])
                datasets[modality]['y'] = np.array(datasets[modality]['y'])
        
        return datasets
    
    def split_multimodal_data(self, datasets, test_size=0.2):
        """Split multimodal data into train and test sets"""
        splits = {}
        
        for modality, data in datasets.items():
            if len(data['X']) > 0:
                X_train, X_test, y_train, y_test = train_test_split(
                    data['X'], data['y'], 
                    test_size=test_size, 
                    random_state=self.config['preprocessing']['random_state'],
                    stratify=data['y']
                )
                
                splits[modality] = {
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test
                }
        
        return splits