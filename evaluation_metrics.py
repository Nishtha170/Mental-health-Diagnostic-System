from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_auc_score, hamming_loss,
    jaccard_score, matthews_corrcoef, log_loss
)
import numpy as np
import pandas as pd

class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def calculate_all_metrics(self, y_true, y_pred, y_pred_proba=None):
        metrics = {}
        
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['Cohens Kappa'] = cohen_kappa_score(y_true, y_pred)
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            metrics['AUC ROC'] = roc_auc_score(y_true, y_pred_proba)
            metrics['Log Loss'] = log_loss(y_true, y_pred_proba)
        else:
            metrics['AUC ROC'] = 0.0
            metrics['Log Loss'] = 0.0
        
        metrics['Hamming Loss'] = hamming_loss(y_true, y_pred)
        metrics['Jaccard Similarity Coeff'] = jaccard_score(y_true, y_pred, average='weighted')
        metrics['Matthews Correlation'] = matthews_corrcoef(y_true, y_pred)
        
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            balanced_acc = recall_score(y_true, y_pred, average='binary')
        else:
            balanced_acc = recall_score(y_true, y_pred, average='weighted')
        metrics['Balanced Accuracy'] = balanced_acc
        
        return metrics
    
    def add_results(self, model_name, dataset_name, metrics):
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][dataset_name] = metrics
    
    def get_results_dataframe(self):
        all_results = []
        for model_name, datasets in self.results.items():
            for dataset_name, metrics in datasets.items():
                row = {'Model': model_name, 'Dataset': dataset_name}
                row.update(metrics)
                all_results.append(row)
        
        return pd.DataFrame(all_results)
    
    def save_results(self, filepath):
        df = self.get_results_dataframe()
        df.to_csv(filepath, index=False)
        return df