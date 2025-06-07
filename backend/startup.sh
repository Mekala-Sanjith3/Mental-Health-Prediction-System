#!/bin/bash
set -e

echo "Starting deployment setup..."

# Train the model if it doesn't exist
if [ ! -f "app/models/best_model.pkl" ]; then
    echo "Model not found, training model..."
    python train_model.py
else
    echo "Model already exists, skipping training..."
fi

# Start the FastAPI server
echo "Starting FastAPI server..."
cd app
exec uvicorn main:app --host 0.0.0.0 --port $PORT 