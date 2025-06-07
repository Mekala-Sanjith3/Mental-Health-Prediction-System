#!/usr/bin/env python3
"""
Test script to verify model loading functionality
"""
import os
import sys
import pickle

def test_model_loading():
    """Test if the model files can be loaded correctly"""
    print("Testing model loading...")
    
    # Try multiple possible model directories
    possible_dirs = [
        "app/models",
        "models",
        "../models"
    ]
    
    models_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            models_dir = dir_path
            break
    
    if models_dir is None:
        print("❌ Models directory not found in any expected location")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir('.')}")
        return False
        
    print(f"✅ Using models directory: {models_dir}")
    
    # Check if all required files exist
    required_files = ["best_model.pkl", "preprocessor.pkl", "encoders.pkl", "feature_names.pkl"]
    missing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(models_dir, file_name)
        if os.path.exists(file_path):
            print(f"✅ Found {file_name}")
        else:
            print(f"❌ Missing {file_name}")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Try to load each file
    try:
        # Load model
        model_path = os.path.join(models_dir, "best_model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        
        # Load preprocessor
        scaler_path = os.path.join(models_dir, "preprocessor.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded successfully: {type(scaler).__name__}")
        
        # Load encoders
        encoders_path = os.path.join(models_dir, "encoders.pkl")
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        print(f"✅ Encoders loaded successfully: {len(encoders)} encoders")
        
        # Load feature names
        features_path = os.path.join(models_dir, "feature_names.pkl")
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"✅ Feature names loaded successfully: {len(feature_names)} features")
        
        print("✅ All model components loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1) 