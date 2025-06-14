import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    try:

        data_paths = [
            '../data/Mental Health Dataset.csv',
            '../../data/Mental Health Dataset.csv',
            'data/Mental Health Dataset.csv',
            'Mental Health Dataset.csv'
        ]
        
        df = None
        for data_path in data_paths:
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                print(f"Loaded dataset from {data_path} with {len(df)} rows and {len(df.columns)} columns")
                break
        
        if df is None:
            print("Dataset not found in any expected location, creating synthetic data for demo...")
            return create_synthetic_data()
        

        df = df.dropna()
        

        required_features = [
            'Gender', 'Country', 'Occupation', 'self_employed', 'family_history',
            'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History',
            'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness',
            'mental_health_interview', 'care_options', 'treatment'
        ]
        

        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            print("Available columns:", list(df.columns))
            return create_synthetic_data()
        
        return df[required_features]
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return create_synthetic_data()

def create_synthetic_data():
    print("Creating synthetic dataset for demo...")
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
        'Country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany'], n_samples),
        'Occupation': np.random.choice(['Software Engineer', 'Teacher', 'Doctor', 'Student', 'Manager'], n_samples),
        'self_employed': np.random.choice(['Yes', 'No'], n_samples),
        'family_history': np.random.choice(['Yes', 'No'], n_samples),
        'Days_Indoors': np.random.choice(['1-14 days', '15-30 days', '31-60 days', 'More than 2 months'], n_samples),
        'Growing_Stress': np.random.choice(['Yes', 'No', 'Maybe'], n_samples),
        'Changes_Habits': np.random.choice(['Yes', 'No', 'Maybe'], n_samples),
        'Mental_Health_History': np.random.choice(['Yes', 'No', 'Maybe'], n_samples),
        'Mood_Swings': np.random.choice(['High', 'Medium', 'Low'], n_samples),
        'Coping_Struggles': np.random.choice(['Yes', 'No'], n_samples),
        'Work_Interest': np.random.choice(['Yes', 'No', 'Maybe'], n_samples),
        'Social_Weakness': np.random.choice(['Yes', 'No', 'Maybe'], n_samples),
        'mental_health_interview': np.random.choice(['Yes', 'No', 'Maybe'], n_samples),
        'care_options': np.random.choice(['Yes', 'No', 'Not sure'], n_samples),
    }
    
    treatment = []
    for i in range(n_samples):
        score = 0
        if data['family_history'][i] == 'Yes': score += 2
        if data['Growing_Stress'][i] == 'Yes': score += 2
        if data['Mental_Health_History'][i] == 'Yes': score += 3
        if data['Mood_Swings'][i] == 'High': score += 2
        if data['Coping_Struggles'][i] == 'Yes': score += 2
        

        score += np.random.randint(-2, 3)
        treatment.append('Yes' if score >= 4 else 'No')
    
    data['treatment'] = treatment
    return pd.DataFrame(data)

def encode_features(df):
    encoders = {}
    encoded_df = df.copy()
    
    categorical_columns = [
        'Gender', 'Country', 'Occupation', 'self_employed', 'family_history',
        'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History',
        'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness',
        'mental_health_interview', 'care_options'
    ]
    
    for col in categorical_columns:
        if col in encoded_df.columns:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            encoders[col] = le
    
    return encoded_df, encoders

def main():
    print("Starting model training...")
    

    possible_models_dirs = ['app/models', 'models', '../app/models']
    models_dir = None
    

    for dir_path in possible_models_dirs:
        if os.path.exists(dir_path):
            models_dir = dir_path
            break
    

    if models_dir is None:
        models_dir = 'app/models'
        os.makedirs(models_dir, exist_ok=True)
    
    print(f"Using models directory: {models_dir}")
    print(f"Models directory exists: {os.path.exists(models_dir)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    

    df = load_and_preprocess_data()
    print(f"Dataset shape: {df.shape}")
    
    encoded_df, encoders = encode_features(df)
    
    X = encoded_df.drop('treatment', axis=1)
    y = LabelEncoder().fit_transform(encoded_df['treatment'])
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with accuracy: {best_score:.4f}")
    

    with open(f'{models_dir}/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(f'{models_dir}/preprocessor.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(f'{models_dir}/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    

    feature_names = list(X.columns)
    with open(f'{models_dir}/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("\nModel training completed!")
    print("Saved files:")
    print(f"- {models_dir}/best_model.pkl")
    print(f"- {models_dir}/preprocessor.pkl")
    print(f"- {models_dir}/encoders.pkl")
    print(f"- {models_dir}/feature_names.pkl")

if __name__ == "__main__":
    main() 