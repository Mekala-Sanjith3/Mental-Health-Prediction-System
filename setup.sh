#!/bin/bash

# Mental Health Prediction System - Development Setup Script

echo "🧠 Mental Health Prediction System Setup"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if Node.js is installed  
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed. Please install Node.js 14+ and try again."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Backend setup
echo "🔧 Setting up backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Backend setup complete"
echo ""

# Frontend setup
cd ../frontend
echo "🎨 Setting up frontend..."

# Install dependencies
echo "Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete"
echo ""

# Create necessary directories
cd ..
mkdir -p data models

echo "🎯 Setup Summary:"
echo "=================="
echo "✅ Backend: Python dependencies installed"
echo "✅ Frontend: Node.js dependencies installed"
echo "✅ Project structure ready"
echo ""
echo "📋 Next Steps:"
echo "1. Add your dataset to the 'data/' directory"
echo "2. Train the model: cd backend && python app/model_training.py"
echo "3. Start backend: cd backend && uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
echo "4. Start frontend: cd frontend && npm start"
echo ""
echo "🌐 Access URLs:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://127.0.0.1:8000"
echo "- API Docs: http://127.0.0.1:8000/docs"
echo ""
echo "🎉 Setup complete! Happy coding!" 