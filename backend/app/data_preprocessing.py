import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class MentalHealthDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'treatment'
        
    def load_data(self, file_path='../data/Mental Health Dataset.csv'):
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded with shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
            
    def handle_missing_values(self, df):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if col == 'self_employed':
                    df[col].fillna('Unknown', inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
                
        return df
        
    def feature_engineering(self, df):
        if 'Timestamp' in df.columns:
            df = df.drop('Timestamp', axis=1)
            
        binary_mappings = {
            'family_history': {'Yes': 1, 'No': 0},
            'treatment': {'Yes': 1, 'No': 0},
            'Growing_Stress': {'Yes': 1, 'No': 0},
            'Changes_Habits': {'Yes': 1, 'No': 0},
            'Mental_Health_History': {'Yes': 1, 'No': 0},
            'Coping_Struggles': {'Yes': 1, 'No': 0},
            'Work_Interest': {'Yes': 1, 'No': 0},
            'Social_Weakness': {'Yes': 1, 'No': 0},
            'mental_health_interview': {'Yes': 1, 'No': 0},
            'self_employed': {'Yes': 1, 'No': 0, 'Unknown': -1}
        }
        
        for col, mapping in binary_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        if 'Mood_Swings' in df.columns:
            mood_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
            df['Mood_Swings'] = df['Mood_Swings'].map(mood_mapping)
            
        if 'Days_Indoors' in df.columns:
            days_mapping = {
                '1-14 days': 1, 
                '15-30 days': 2, 
                '31-60 days': 3, 
                'More than 2 months': 4,
                'Go out Every day': 0
            }
            df['Days_Indoors'] = df['Days_Indoors'].map(days_mapping)
            
        if 'care_options' in df.columns:
            care_mapping = {'Yes': 1, 'No': 0, 'Not sure': 0.5}
            df['care_options'] = df['care_options'].map(care_mapping)
            
        return df
        
    def encode_categorical_features(self, df):
        categorical_cols = ['Gender', 'Country', 'Occupation']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
        return df
        
    def scale_features(self, X_train, X_test=None):
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
        
    def preprocess_data(self, file_path='../data/Mental Health Dataset.csv'):
        df = self.load_data(file_path)
        if df is None:
            return None, None, None, None
            
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        df = self.handle_missing_values(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical_features(df)
        
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        self.feature_columns = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    def save_preprocessor(self, filepath='../models/preprocessor.pkl'):
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        print(f"Preprocessor saved to {filepath}")
        
    def load_preprocessor(self, filepath='../models/preprocessor.pkl'):
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        self.feature_columns = preprocessor_data['feature_columns']
        print(f"Preprocessor loaded from {filepath}")
        
    def transform_single_input(self, input_data):
        df = pd.DataFrame([input_data])
        
        df = self.handle_missing_values(df)
        df = self.feature_engineering(df)
        
        for col in ['Gender', 'Country', 'Occupation']:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError:
                    df[col] = 0
        
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_columns]
        scaled_data = self.scaler.transform(df)
        
        return scaled_data

if __name__ == "__main__":
    preprocessor = MentalHealthDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
    
    if X_train is not None:
        preprocessor.save_preprocessor()
        print("Preprocessing completed successfully!")
