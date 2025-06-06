# Mental Health Treatment Prediction System

A comprehensive machine learning application that predicts mental health treatment requirements based on survey responses. Built with React, FastAPI, and advanced ML models for accurate mental health assessments.

## 🚀 Live Demo
[Add your deployed app URL here when available]

## 👨‍💻 Developer
**Mekala Maria Sanjith Reddy**

## 🌟 Features

- **Interactive Web Interface**: Multi-step form with validation
- **Machine Learning Predictions**: Random Forest model trained on survey data
- **Analysis Dashboard**: Confidence scores, feature importance, and visualizations
- **RESTful API**: FastAPI backend with authentication
- **Modern Tech Stack**: React + Material-UI, FastAPI, scikit-learn
- **Production Ready**: Docker support and scalable architecture

## 📊 What It Predicts

The system analyzes various factors to predict whether mental health treatment may be beneficial:

- Personal demographics (gender, country, occupation)
- Employment status and work interest
- Mental health history (personal and family)
- Current symptoms and stress levels
- Coping mechanisms and social factors
- Awareness of care options

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mental-health-prediction.git
   cd mental-health-prediction
   ```

2. **Set up the backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   
   # Train the model (first time only)
   python app/model_training.py
   
   # Start the API server
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

3. **Set up the frontend**
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - API Documentation: http://127.0.0.1:8000/docs
   - Health Check: http://127.0.0.1:8000/health

## 🏗️ Project Structure

```
mental-health-prediction/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # Main FastAPI application
│   │   ├── data_preprocessing.py  # Data preprocessing pipeline
│   │   └── model_training.py      # ML model training
│   ├── tests/              # Backend tests
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── public/            # Static files
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API services
│   │   └── App.js         # Main app component
│   └── package.json       # Node.js dependencies
├── data/                  # Dataset (not included in git)
├── models/                # Trained models (not included in git)
├── .gitignore
└── README.md
```

## 🔧 Usage

### Web Interface

1. Navigate to http://localhost:3000
2. Fill out the three-step assessment form:
   - **Step 1**: Personal Information
   - **Step 2**: Mental Health History  
   - **Step 3**: Current Status
3. Click "Get Prediction" to receive results
4. View detailed analysis including:
   - Treatment recommendation
   - Confidence level
   - Key contributing factors
   - Feature importance analysis

### API Usage

The backend provides a RESTful API with the following endpoints:

- `GET /health` - Health check
- `POST /predict` - Get prediction (requires authentication)
- `GET /model-info` - Model information
- `GET /docs` - Interactive API documentation

Example API call:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Authorization: Bearer demo-token-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Country": "USA",
    "Occupation": "Developer",
    "self_employed": "No",
    "family_history": "No",
    "Days_Indoors": "1-14 days",
    "Growing_Stress": "Yes",
    "Changes_Habits": "Yes",
    "Mental_Health_History": "No",
    "Mood_Swings": "Medium",
    "Coping_Struggles": "Yes",
    "Work_Interest": "Yes",
    "Social_Weakness": "No",
    "mental_health_interview": "No",
    "care_options": "Yes"
  }'
```

## 🧪 Machine Learning Pipeline

### Data Preprocessing
- Handles missing values intelligently
- Feature engineering for binary and ordinal variables
- Label encoding for categorical features
- StandardScaler for numerical normalization

### Model Training
- Random Forest Classifier as the primary model
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust evaluation
- Feature importance analysis

### Model Performance
- Accuracy: ~74%
- Precision: ~73%
- Recall: ~78%
- F1-Score: ~75%
- ROC-AUC: ~80%

## 🔒 Security

- Token-based authentication for API endpoints
- Input validation and sanitization
- CORS configuration for secure cross-origin requests
- Environment variable configuration for sensitive data

## 🧪 Testing

Run the test suite:

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend
npm test
```

## 📈 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified mental health professionals for any mental health concerns.

## 📞 Support

For support, please open an issue on GitHub or contact [2300031810cseh1@gmail.com].

---

**Built with ❤️ for mental health awareness and accessibility.**

