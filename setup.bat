@echo off
echo.
echo 🧠 Mental Health Prediction System Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 3 is required but not installed. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is required but not installed. Please install Node.js 14+ and try again.
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed
echo.

REM Backend setup
echo 🔧 Setting up backend...
cd backend

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ✅ Backend setup complete
echo.

REM Frontend setup
cd ..\frontend
echo 🎨 Setting up frontend...

REM Install dependencies
echo Installing Node.js dependencies...
npm install

echo ✅ Frontend setup complete
echo.

REM Create necessary directories
cd ..
if not exist "data" mkdir data
if not exist "models" mkdir models

echo 🎯 Setup Summary:
echo ==================
echo ✅ Backend: Python dependencies installed
echo ✅ Frontend: Node.js dependencies installed
echo ✅ Project structure ready
echo.
echo 📋 Next Steps:
echo 1. Add your dataset to the 'data/' directory
echo 2. Train the model: cd backend ^&^& python app/model_training.py
echo 3. Start backend: cd backend ^&^& uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
echo 4. Start frontend: cd frontend ^&^& npm start
echo.
echo 🌐 Access URLs:
echo - Frontend: http://localhost:3000
echo - Backend API: http://127.0.0.1:8000
echo - API Docs: http://127.0.0.1:8000/docs
echo.
echo 🎉 Setup complete! Happy coding!
echo.
pause 