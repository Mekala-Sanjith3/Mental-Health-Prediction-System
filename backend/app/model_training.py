import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MentalHealthModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.results = {}
        
    def load_data(self):
        try:
            X_train = np.load('../models/X_train.npy')
            X_test = np.load('../models/X_test.npy')
            y_train = np.load('../models/y_train.npy')
            y_test = np.load('../models/y_test.npy')
            
            with open('../models/preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
                self.feature_names = preprocessor['feature_columns']
            
            print(f"Data loaded: Training {X_train.shape}, Test {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except FileNotFoundError:
            print("Preprocessed data not found. Run preprocessing first.")
            return None, None, None, None
            
    def initialize_models(self):
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        self.models['RandomForest'] = {
            'model': RandomForestClassifier(random_state=42),
            'params': rf_param_grid,
            'search_type': 'random'
        }
        
        lr_param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        }
        
        self.models['LogisticRegression'] = {
            'model': LogisticRegression(random_state=42),
            'params': lr_param_grid,
            'search_type': 'grid'
        }
        
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        self.models['XGBoost'] = {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': xgb_param_grid,
            'search_type': 'random'
        }
        
    def train_model(self, model_name, X_train, y_train, cv=5):
        print(f"Training {model_name}...")
        
        model_config = self.models[model_name]
        base_model = model_config['model']
        param_grid = model_config['params']
        search_type = model_config['search_type']
        
        if search_type == 'grid':
            search = GridSearchCV(
                base_model, param_grid, cv=cv, 
                scoring='f1', n_jobs=-1, verbose=0
            )
        else:
            search = RandomizedSearchCV(
                base_model, param_grid, cv=cv, 
                scoring='f1', n_jobs=-1, verbose=0,
                n_iter=20, random_state=42
            )
        
        search.fit(X_train, y_train)
        
        print(f"{model_name} - Best CV score: {search.best_score_:.4f}")
        return search.best_estimator_
        
    def evaluate_model(self, model, model_name, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.results[model_name] = metrics
        
        print(f"{model_name} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
        
    def plot_model_comparison(self):
        if not self.results:
            print("No results to plot.")
            return
            
        df_results = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            df_results[metric].plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig('../models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def select_best_model(self):
        if not self.results:
            print("No models trained yet.")
            return None
            
        best_f1 = 0
        best_model_name = None
        
        for model_name, metrics in self.results.items():
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_model_name = model_name
                
        self.best_model_name = best_model_name
        print(f"Best model: {best_model_name} (F1: {best_f1:.4f})")
        return best_model_name
        
    def generate_feature_importance(self, model, model_name):
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                print(f"Feature importance not available for {model_name}")
                return
            
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            top_features = feature_importance_df.head(15)
            sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
            plt.title(f'Top 15 Feature Importances - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'../models/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_importance_df
            
        except Exception as e:
            print(f"Error generating feature importance: {e}")
            return None
    
    def save_best_model(self):
        if self.best_model is None:
            print("No best model selected.")
            return
            
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'accuracy': self.results[self.best_model_name]['accuracy'],
            'f1_score': self.results[self.best_model_name]['f1_score']
        }
        
        with open('../models/best_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Best model saved: {self.best_model_name}")
        
    def train_all_models(self):
        X_train, X_test, y_train, y_test = self.load_data()
        if X_train is None:
            return
            
        self.initialize_models()
        
        trained_models = {}
        
        for model_name in self.models.keys():
            model = self.train_model(model_name, X_train, y_train)
            trained_models[model_name] = model
            self.evaluate_model(model, model_name, X_test, y_test)
            
        best_model_name = self.select_best_model()
        if best_model_name:
            self.best_model = trained_models[best_model_name]
            self.save_best_model()
            self.generate_feature_importance(self.best_model, best_model_name)
            
        self.plot_model_comparison()
        
        print("\nTraining completed!")
        print(f"Best model: {self.best_model_name}")
        print("Model saved to '../models/best_model.pkl'")

if __name__ == "__main__":
    trainer = MentalHealthModelTrainer()
    trainer.train_all_models() 