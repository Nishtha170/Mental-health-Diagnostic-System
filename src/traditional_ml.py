from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import yaml

class TraditionalML:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.vectorizers = {}
    
    def train_random_forest(self, X_train_tfidf, y_train):
        model = RandomForestClassifier(
            n_estimators=self.config['models']['random_forest']['n_estimators'],
            criterion=self.config['models']['random_forest']['criterion'],
            max_depth=self.config['models']['random_forest']['max_depth'],
            random_state=self.config['models']['random_forest']['random_state']
        )
        
        model.fit(X_train_tfidf, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_svc(self, X_train_tfidf, y_train):
        model = LinearSVC(
            C=self.config['models']['svc']['C'],
            loss=self.config['models']['svc']['loss'],
            random_state=self.config['models']['svc']['random_state']
        )
        
        model.fit(X_train_tfidf, y_train)
        self.models['svc'] = model
        return model
    
    def evaluate_model(self, model_name, X_test_tfidf, y_test):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        predictions = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions
        }
    
    def save_model(self, model_name, path):
        if model_name in self.models:
            joblib.dump(self.models[model_name], path)