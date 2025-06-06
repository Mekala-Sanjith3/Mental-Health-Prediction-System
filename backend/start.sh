#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p models

# Check if model files exist, if not, train the model
if [ ! -f "models/best_model.pkl" ] || [ ! -f "models/preprocessor.pkl" ]; then
    echo "Models not found. Training model..."
    
    # Check if data file exists
    if [ ! -f "data/Mental Health Dataset.csv" ]; then
        echo "Error: Mental Health Dataset.csv not found in data directory"
        echo "Please ensure the dataset is available for training"
        exit 1
    fi
    
    # Run preprocessing
    echo "Running data preprocessing..."
    cd app
    python data_preprocessing.py
    
    # Run model training
    echo "Training ML models..."
    python model_training.py
    
    cd ..
    echo "Model training completed!"
else
    echo "Models found. Skipping training."
fi

# Start the FastAPI server
echo "Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 