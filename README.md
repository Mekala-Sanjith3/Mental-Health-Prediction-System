# Mental Health Treatment Prediction System

A comprehensive machine learning-powered web application that predicts mental health treatment requirements based on user responses to a detailed assessment questionnaire.

## 🌟 Features

- **Intelligent Assessment**: Multi-step questionnaire covering personal information, mental health history, and current status
- **ML-Powered Predictions**: Uses RandomForest classifier with 73%+ accuracy
- **Comprehensive Analysis**: Provides detailed risk assessment, confidence metrics, and personalized recommendations
- **Professional UI**: Modern, responsive design built with React and Material-UI
- **Real-time Results**: Instant predictions with detailed visualizations
- **Educational Resources**: Includes mental health resources and support information

## 🏗️ Architecture

### Frontend
- **Framework**: React 18
- **UI Library**: Material-UI (MUI)
- **Deployment**: Vercel
- **Features**: Progressive web app, responsive design, interactive charts

### Backend
- **Framework**: FastAPI
- **ML Library**: Scikit-learn
- **Deployment**: Render (Docker)
- **Features**: RESTful API, token authentication, comprehensive health monitoring

### Machine Learning
- **Algorithm**: RandomForest Classifier
- **Accuracy**: 73.28%
- **Features**: 15 input parameters
- **Output**: Binary classification with confidence scores and detailed analysis

## 🚀 Live Demo

- **Frontend**: [https://mental-health-prediction-system.vercel.app](https://mental-health-prediction-system.vercel.app)
- **Backend API**: [https://mental-health-prediction-system-wj25.onrender.com](https://mental-health-prediction-system-wj25.onrender.com)
- **API Documentation**: [https://mental-health-prediction-system-wj25.onrender.com/docs](https://mental-health-prediction-system-wj25.onrender.com/docs)

## 📋 Prerequisites

- Node.js 16+ (for frontend)
- Python 3.9+ (for backend)
- Git

## 🛠️ Local Development Setup

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mekala-Sanjith3/Mental-Health-Prediction-System.git
   cd Mental-Health-Prediction-System/backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python train_model.py
   ```

5. **Start the server**
   ```bash
   cd app
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set environment variables**
   ```bash
   # Create .env file
   echo "REACT_APP_API_URL=http://localhost:8000" > .env
   ```

4. **Start development server**
   ```bash
   npm start
   ```

The application will be available at `http://localhost:3000`

## 📊 Model Information

### Dataset Features
- **Gender**: Male, Female, Other
- **Country**: Geographic location
- **Occupation**: Professional field
- **Self Employment**: Employment status
- **Family History**: Mental health history in family
- **Days Indoors**: Social isolation indicator
- **Growing Stress**: Stress level changes
- **Changes in Habits**: Behavioral modifications
- **Mental Health History**: Personal mental health background
- **Mood Swings**: Emotional stability indicator
- **Coping Struggles**: Ability to handle challenges
- **Work Interest**: Professional engagement level
- **Social Weakness**: Social interaction difficulties
- **Mental Health Interview**: Openness to discussion
- **Care Options**: Awareness of available resources

### Model Performance
- **Algorithm**: RandomForest Classifier
- **Training Accuracy**: 73.28%
- **Features**: 15 input parameters
- **Cross-validation**: Implemented
- **Preprocessing**: Label encoding, standard scaling

## 🔧 API Endpoints

### Authentication
All prediction endpoints require Bearer token authentication.
- **Token**: `demo-token-12345`

### Main Endpoints

#### Health Check
```http
GET /health
```
Returns API status and model loading state.

#### Prediction
```http
POST /predict
Authorization: Bearer demo-token-12345
Content-Type: application/json

{
  "Gender": "Male",
  "Country": "USA",
  "Occupation": "Software Engineer",
  "self_employed": "No",
  "family_history": "Yes",
  "Days_Indoors": "15-30 days",
  "Growing_Stress": "Yes",
  "Changes_Habits": "Yes",
  "Mental_Health_History": "No",
  "Mood_Swings": "High",
  "Coping_Struggles": "Yes",
  "Work_Interest": "Yes",
  "Social_Weakness": "No",
  "mental_health_interview": "Yes",
  "care_options": "Yes"
}
```

#### Model Information
```http
GET /model-info
Authorization: Bearer demo-token-12345
```

## 🎨 UI Components

### Assessment Form
- **Multi-step wizard**: 3-step process for better UX
- **Form validation**: Real-time validation with error messages
- **Progress indicator**: Visual progress tracking
- **Responsive design**: Works on all device sizes

### Results Display
- **Prediction summary**: Clear treatment recommendation
- **Confidence analysis**: Visual confidence metrics with charts
- **Risk assessment**: Detailed risk level analysis
- **Personalized recommendations**: Tailored advice based on input
- **Educational content**: Mental health resources and support information

## 🔒 Security Features

- **Token-based authentication**: Secure API access
- **Input validation**: Comprehensive data validation
- **CORS configuration**: Secure cross-origin requests
- **Error handling**: Graceful error management

## 📱 Responsive Design

The application is fully responsive and works seamlessly across:
- **Desktop**: Full-featured experience
- **Tablet**: Optimized layout
- **Mobile**: Touch-friendly interface

## 🚀 Deployment

### Frontend (Vercel)
- Automatic deployment from GitHub
- Environment variables configured
- CDN distribution for optimal performance

### Backend (Render)
- Docker-based deployment
- Automatic model training during build
- Health monitoring and auto-restart

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mekala Maria Sanjith Reddy**

- GitHub: [@Mekala-Sanjith3](https://github.com/Mekala-Sanjith3)
- LinkedIn: [Mekala Maria Sanjith Reddy](https://linkedin.com/in/mekala-sanjith)

## 🙏 Acknowledgments

- Mental health dataset contributors
- Open source community
- Material-UI team for the excellent component library
- FastAPI team for the robust framework

## 📞 Support

If you have any questions or need support, please:
1. Check the [API documentation](https://mental-health-prediction-system-wj25.onrender.com/docs)
2. Open an issue on GitHub
3. Contact the author

---

**⚠️ Disclaimer**: This application is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions regarding mental health conditions.

