import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from pathlib import Path
import re

def create_directories():
    """Create all required directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models/saved_models', 
        'models/training_logs',
        'results/tables',
        'results/figures',
        'results/reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def generate_realistic_dataset(dataset_type, num_samples=800):
    """Generate more realistic mental health datasets with noise and ambiguity"""
    
    # More diverse and realistic phrases
    mental_health_phrases = [
        "I've been feeling really down lately and can't seem to shake it",
        "My anxiety has been through the roof these past few weeks",
        "Sometimes I feel so hopeless about everything",
        "The depression makes it hard to get out of bed most days",
        "I keep having panic attacks and don't know why",
        "Mental health has been a real struggle for me recently",
        "Feeling overwhelmed by everything going on in life",
        "The sadness just won't go away no matter what I try",
        "I'm really struggling with my thoughts and emotions",
        "Some days are better than others but mostly it's hard",
        "Went to therapy today and talked about some tough stuff",
        "Medication helps but I still have really bad days",
        "Feeling isolated even when I'm around other people",
        "The stress from work is really affecting my mental state",
        "I worry constantly about things that might never happen"
    ]
    
    neutral_phrases = [
        "Had a pretty good day at work today",
        "Looking forward to the weekend with friends",
        "The weather has been really nice lately",
        "Just finished a great book I was reading",
        "Planning a trip for the upcoming holidays",
        "Work has been busy but manageable recently",
        "Enjoying some quiet time at home this evening",
        "Learning new skills for my personal development",
        "Spent some quality time with family today",
        "Trying out new recipes in the kitchen lately",
        "Getting back into my exercise routine again",
        "Caught up with some old friends recently",
        "Working on some home improvement projects",
        "Enjoying my hobbies in my free time",
        "Life has been pretty normal and routine lately"
    ]
    
    toxic_phrases = [
        "I'm so angry at everything and everyone around me",
        "People can be so stupid and frustrating sometimes",
        "This situation is completely ridiculous and unfair",
        "I hate how things are going in my life right now",
        "Everything seems pointless and not worth the effort",
        "I'm fed up with all the nonsense happening lately",
        "This is the worst possible outcome I could imagine",
        "I can't believe how terrible everything has become",
        "People are just awful and disappointing generally",
        "Nothing ever goes right no matter what I try to do"
    ]
    
    texts = []
    labels = []
    
    for i in range(num_samples):
        if dataset_type == 'dataset1':
            # Binary classification: mental health vs neutral with more ambiguity
            if np.random.random() > 0.5:
                base_phrase = np.random.choice(mental_health_phrases)
                # Add some neutral words to create ambiguity
                if np.random.random() > 0.3:
                    neutral_words = np.random.choice(neutral_phrases).split()[:3]
                    base_phrase += " " + " ".join(neutral_words)
                labels.append(1)
            else:
                base_phrase = np.random.choice(neutral_phrases)
                # Add some mental health related words to create ambiguity
                if np.random.random() > 0.3:
                    mental_words = np.random.choice(mental_health_phrases).split()[:2]
                    base_phrase += " " + " ".join(mental_words)
                labels.append(0)
                
        elif dataset_type == 'dataset2':
            # Binary classification: toxic vs neutral with more realistic distribution
            if np.random.random() > 0.8:  # Toxic comments are rarer
                base_phrase = np.random.choice(toxic_phrases)
                labels.append(1)
            else:
                base_phrase = np.random.choice(neutral_phrases)
                # Sometimes add mild negative sentiment
                if np.random.random() > 0.7:
                    base_phrase = "Not great but " + base_phrase.lower()
                labels.append(0)
                
        elif dataset_type == 'dataset3':
            # Multi-class classification with more realistic overlap
            conditions = ['stress', 'depression', 'bipolar', 'personality_disorder', 'anxiety']
            condition_idx = np.random.randint(0, 5)
            
            # Create overlapping symptoms
            base_symptoms = [
                "trouble sleeping", "changes in appetite", "low energy", 
                "difficulty concentrating", "mood changes", "feeling overwhelmed"
            ]
            
            condition_specific = {
                0: ["work pressure", "deadlines", "too much to do", "burnout"],
                1: ["deep sadness", "loss of interest", "hopelessness", "empty feeling"],
                2: ["mood swings", "high energy periods", "low energy periods", "irritability"],
                3: ["relationship issues", "identity confusion", "emotional instability", "fear of abandonment"],
                4: ["constant worry", "panic feelings", "restlessness", "physical tension"]
            }
            
            # Mix general symptoms with condition-specific ones
            general_symptoms = np.random.choice(base_symptoms, size=2, replace=False)
            specific_symptoms = np.random.choice(condition_specific[condition_idx], size=2, replace=False)
            all_symptoms = list(general_symptoms) + list(specific_symptoms)
            np.random.shuffle(all_symptoms)
            
            base_phrase = f"Experiencing {', '.join(all_symptoms)} lately"
            labels.append(condition_idx)
        
        # Add realistic variations and noise
        variations = [
            "tbh", "imo", "idk", "smh", "fr", "ngl", "lowkey", "highkey",
            "anyways", "tbf", "afaik", "rn", "atm"
        ]
        
        # Add some random variations
        if np.random.random() > 0.7:
            base_phrase += " " + np.random.choice(variations)
        
        # Add typos and informal language occasionally
        if np.random.random() > 0.8:
            base_phrase = base_phrase.replace('ing', 'in').replace('you', 'u').replace('are', 'r')
        
        texts.append(base_phrase)
    
    return pd.DataFrame({'text': texts, 'label': labels})

def create_all_datasets():
    """Create all three datasets with realistic data"""
    print("Generating realistic synthetic datasets...")
    
    dataset1 = generate_realistic_dataset('dataset1', 1000)
    dataset2 = generate_realistic_dataset('dataset2', 1000)
    dataset3 = generate_realistic_dataset('dataset3', 1000)
    
    dataset1.to_csv('data/raw/dataset1_reddit_mental_health.csv', index=False)
    dataset2.to_csv('data/raw/dataset2_toxic_comments.csv', index=False)
    dataset3.to_csv('data/raw/dataset3_multiclass_mental_health.csv', index=False)
    
    print(f"Dataset 1 shape: {dataset1.shape}")
    print(f"Dataset 2 shape: {dataset2.shape}")
    print(f"Dataset 3 shape: {dataset3.shape}")
    print("Realistic datasets saved successfully!")

def build_lstm_model(vocab_size, num_classes, max_length=100):
    """Build LSTM model with proper architecture for the task"""
    model = Sequential()
    
    model.add(Embedding(vocab_size, 128, input_length=max_length))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    
    if num_classes == 2:
        # Binary classification
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        # Multi-class classification
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=metrics
    )
    
    return model

def train_and_evaluate_models():
    """Train and evaluate models on the datasets"""
    print("Training and evaluating models...")
    
    results = {}
    
    for dataset_name in ['dataset1', 'dataset2', 'dataset3']:
        print(f"\n=== Processing {dataset_name} ===")
        
        # Load dataset
        df = pd.read_csv(f'data/raw/{dataset_name}_reddit_mental_health.csv' if dataset_name == 'dataset1' else 
                        f'data/raw/{dataset_name}_toxic_comments.csv' if dataset_name == 'dataset2' else 
                        f'data/raw/{dataset_name}_multiclass_mental_health.csv')
        
        # Basic text cleaning
        df['cleaned_text'] = df['text'].str.lower().str.replace(r'[^\w\s]', '').str.replace(r'\d+', '')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )
        
        dataset_results = {}
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Random Forest with regularization
        print("  Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf_model.fit(X_train_tfidf, y_train)
        rf_pred = rf_model.predict(X_test_tfidf)
        dataset_results['Random Forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, rf_pred, average='weighted', zero_division=0)
        }
        
        # SVM with regularization
        print("  Training SVM...")
        svm_model = LinearSVC(random_state=42, C=0.5, max_iter=1000)
        svm_model.fit(X_train_tfidf, y_train)
        svm_pred = svm_model.predict(X_test_tfidf)
        dataset_results['SVM'] = {
            'accuracy': accuracy_score(y_test, svm_pred),
            'precision': precision_score(y_test, svm_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, svm_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, svm_pred, average='weighted', zero_division=0)
        }
        
        # LSTM with proper architecture
        print("  Training LSTM...")
        try:
            tokenizer = Tokenizer(num_words=1000)
            tokenizer.fit_on_texts(X_train)
            X_train_seq = tokenizer.texts_to_sequences(X_train)
            X_test_seq = tokenizer.texts_to_sequences(X_test)
            
            X_train_pad = pad_sequences(X_train_seq, maxlen=100)
            X_test_pad = pad_sequences(X_test_seq, maxlen=100)
            
            num_classes = len(np.unique(y_train))
            
            lstm_model = build_lstm_model(1000, num_classes)
            
            history = lstm_model.fit(
                X_train_pad, y_train,
                epochs=5,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Predict based on model type
            if num_classes == 2:
                lstm_pred_proba = lstm_model.predict(X_test_pad, verbose=0)
                lstm_pred = (lstm_pred_proba > 0.5).astype(int).flatten()
            else:
                lstm_pred_proba = lstm_model.predict(X_test_pad, verbose=0)
                lstm_pred = np.argmax(lstm_pred_proba, axis=1)
            
            dataset_results['LSTM'] = {
                'accuracy': accuracy_score(y_test, lstm_pred),
                'precision': precision_score(y_test, lstm_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, lstm_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, lstm_pred, average='weighted', zero_division=0)
            }
        except Exception as e:
            print(f"    LSTM training error: {e}")
            dataset_results['LSTM'] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        results[dataset_name] = dataset_results
    
    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for dataset_name, models in results.items():
        print(f"\n{dataset_name.upper()}:")
        print("-" * 40)
        for model_name, metrics in models.items():
            print(f"  {model_name:15} | Accuracy: {metrics['accuracy']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
                  f"F1-Score: {metrics['f1_score']:.4f}")
    
    # Save results to CSV
    results_list = []
    for dataset_name, models in results.items():
        for model_name, metrics in models.items():
            results_list.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score']
            })
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('results/tables/final_results.csv', index=False)
    print(f"\nResults saved to: results/tables/final_results.csv")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Mental Health Analysis using ML/DL')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic datasets')
    parser.add_argument('--train-models', action='store_true', help='Train all models')
    
    args = parser.parse_args()
    
    # Create directories first
    create_directories()
    
    if args.generate_data:
        create_all_datasets()
    
    if args.train_models:
        results_df = train_and_evaluate_models()
        print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()