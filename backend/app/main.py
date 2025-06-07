from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os
import logging
from datetime import datetime
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Treatment Prediction API",
    description="API for predicting mental health treatment requirements using machine learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Simple token-based authentication (for demo purposes)
VALID_TOKENS = {"demo-token-12345"}  # In production, use proper JWT tokens

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the authentication token"""
    if credentials.credentials not in VALID_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Global variables to store model and preprocessor
model_data = None
preprocessor = None

class MentalHealthInput(BaseModel):
    """Input schema for mental health prediction"""
    Gender: str = Field(..., description="Gender")
    Country: str = Field(..., description="Country")
    Occupation: str = Field(..., description="Occupation")
    self_employed: str = Field(..., description="Self employed status")
    family_history: str = Field(..., description="Family history of mental health")
    Days_Indoors: str = Field(..., description="Days spent indoors")
    Growing_Stress: str = Field(..., description="Growing stress levels")
    Changes_Habits: str = Field(..., description="Changes in habits")
    Mental_Health_History: str = Field(..., description="Mental health history")
    Mood_Swings: str = Field(..., description="Mood swings frequency")
    Coping_Struggles: str = Field(..., description="Coping struggles")
    Work_Interest: str = Field(..., description="Interest in work")
    Social_Weakness: str = Field(..., description="Social weakness")
    mental_health_interview: str = Field(..., description="Mental health interview willingness")
    care_options: str = Field(..., description="Awareness of care options")
    
    class Config:
        # This ensures field names are preserved exactly as defined
        allow_population_by_field_name = True

class RiskAssessment(BaseModel):
    level: str = Field(..., description="Risk level (Low, Moderate, High, Critical)")
    score: float = Field(..., description="Risk score from 0-1")
    factors: List[str] = Field(..., description="Key risk factors identified")
    protective_factors: List[str] = Field(..., description="Protective factors present")

class PersonalizedRecommendation(BaseModel):
    category: str = Field(..., description="Category of recommendation")
    priority: str = Field(..., description="Priority level")
    action: str = Field(..., description="Recommended action")
    description: str = Field(..., description="Detailed description")
    resources: List[str] = Field(..., description="Relevant resources")

class ConfidenceMetrics(BaseModel):
    overall: float = Field(..., description="Overall prediction confidence")
    level: str = Field(..., description="Confidence level description")
    model_certainty: str = Field(..., description="Model certainty explanation")
    prediction_strength: str = Field(..., description="Strength of prediction")

class DetailedAnalysis(BaseModel):
    primary_concerns: List[str] = Field(..., description="Primary mental health concerns")
    contributing_factors: List[str] = Field(..., description="Contributing factors")
    positive_indicators: List[str] = Field(..., description="Positive mental health indicators")
    areas_of_focus: List[str] = Field(..., description="Areas requiring attention")

class EnhancedPredictionResponse(BaseModel):
    prediction: int = Field(..., description="Prediction result (0 or 1)")
    prediction_label: str = Field(..., description="Human readable prediction")
    confidence_metrics: ConfidenceMetrics = Field(..., description="Detailed confidence analysis")
    risk_assessment: RiskAssessment = Field(..., description="Comprehensive risk evaluation")
    detailed_analysis: DetailedAnalysis = Field(..., description="In-depth analysis of factors")
    feature_importance: Dict[str, float] = Field(..., description="Feature importances")
    personalized_recommendations: List[PersonalizedRecommendation] = Field(..., description="Tailored recommendations")
    educational_content: List[str] = Field(..., description="Educational resources")
    support_resources: Dict[str, str] = Field(..., description="Support contact information")
    timestamp: str = Field(..., description="Prediction timestamp")
    session_id: str = Field(..., description="Unique session identifier")

class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    timestamp: str
    model_loaded: bool
    version: str

def load_model():
    """Load the trained model and preprocessor"""
    global model_data, preprocessor
    
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple possible model directories
        possible_dirs = [
            os.path.join(current_dir, "models"),
            os.path.join(os.path.dirname(current_dir), "models"),
            "models"
        ]
        
        models_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                models_dir = dir_path
                break
        
        if models_dir is None:
            logger.error("Models directory not found in any expected location")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Script directory: {current_dir}")
            return
            
        logger.info(f"Using models directory: {models_dir}")
        
        # Load model
        model_path = os.path.join(models_dir, "best_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
            if os.path.exists(models_dir):
                logger.info(f"Models directory contents: {os.listdir(models_dir)}")
            
        # Load individual components
        scaler_path = os.path.join(models_dir, "preprocessor.pkl")
        encoders_path = os.path.join(models_dir, "encoders.pkl")
        features_path = os.path.join(models_dir, "feature_names.pkl")
        
        preprocessor = {}
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                preprocessor['scaler'] = pickle.load(f)
            logger.info(f"Scaler loaded successfully from {scaler_path}")
        else:
            logger.warning(f"Scaler file not found at {scaler_path}")
        
        if os.path.exists(encoders_path):
            with open(encoders_path, 'rb') as f:
                preprocessor['label_encoders'] = pickle.load(f)
            logger.info(f"Encoders loaded successfully from {encoders_path}")
        else:
            logger.warning(f"Encoders file not found at {encoders_path}")
            
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                preprocessor['feature_columns'] = pickle.load(f)
            logger.info(f"Feature names loaded successfully from {features_path}")
        else:
            logger.warning(f"Feature names file not found at {features_path}")
            
    except Exception as e:
        logger.error(f"Error loading model/preprocessor: {e}")

def preprocess_input(input_data: MentalHealthInput) -> np.ndarray:
    """Preprocess input data for prediction"""
    if preprocessor is None:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded")
    
    # Convert input to dictionary
    data_dict = input_data.dict()
    
    # Convert to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Handle missing values (similar to training preprocessing)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            if col == 'self_employed':
                df[col].fillna('Unknown', inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
    
    # Apply feature engineering (similar to training)
    binary_mappings = {
        'family_history': {'Yes': 1, 'No': 0},
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
    
    # Handle ordinal features
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
    
    # Encode categorical features
    categorical_cols = ['Gender', 'Country', 'Occupation']
    for col in categorical_cols:
        if col in df.columns and col in preprocessor['label_encoders']:
            le = preprocessor['label_encoders'][col]
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                # Handle unseen categories
                df[col] = 0
    
    # Ensure all feature columns are present
    feature_columns = preprocessor['feature_columns']
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[feature_columns]
    
    # Scale the features
    scaled_data = preprocessor['scaler'].transform(df)
    
    return scaled_data

def get_feature_importance(model, feature_names: List[str], input_data: np.ndarray) -> Dict[str, float]:
    """Get feature importance for the prediction"""
    try:
        # For tree-based models
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance.tolist()))
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            return dict(list(sorted_importance.items())[:10])
        # For linear models
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            feature_importance = {
                feature_names[i]: float(importance[i]) 
                for i in np.argsort(importance)[-5:]
            }
            return feature_importance
        else:
            return {}
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return {}

def generate_risk_assessment(input_data: MentalHealthInput, prediction: int, confidence: float, feature_importance: Dict[str, float]) -> RiskAssessment:
    risk_factors = []
    protective_factors = []
    
    # Analyze input for risk factors
    if input_data.family_history == "Yes":
        risk_factors.append("Family history of mental health issues")
    if input_data.Mental_Health_History == "Yes":
        risk_factors.append("Personal history of mental health concerns")
    if input_data.Growing_Stress == "Yes":
        risk_factors.append("Increasing stress levels")
    if input_data.Coping_Struggles == "Yes":
        risk_factors.append("Difficulty coping with daily challenges")
    if input_data.Changes_Habits == "Yes":
        risk_factors.append("Recent changes in daily habits")
    if input_data.Social_Weakness == "Yes":
        risk_factors.append("Social interaction difficulties")
    if input_data.Days_Indoors in ["31-60 days", "More than 2 months"]:
        risk_factors.append("Extended periods of social isolation")
    if input_data.Mood_Swings == "High":
        risk_factors.append("Frequent mood fluctuations")
    
    # Analyze protective factors
    if input_data.Work_Interest == "Yes":
        protective_factors.append("Maintained interest in work activities")
    if input_data.care_options == "Yes":
        protective_factors.append("Awareness of available mental health resources")
    if input_data.mental_health_interview == "Yes":
        protective_factors.append("Openness to discussing mental health")
    if input_data.Days_Indoors == "Go out Every day":
        protective_factors.append("Regular social engagement and outdoor activities")
    if input_data.family_history == "No" and input_data.Mental_Health_History == "No":
        protective_factors.append("No significant mental health history")
    
    # Determine risk level
    if prediction == 1 and confidence >= 0.8:
        level = "High"
        score = 0.8 + (confidence - 0.8) * 0.2 / 0.2
    elif prediction == 1 and confidence >= 0.6:
        level = "Moderate"
        score = 0.6 + (confidence - 0.6) * 0.2 / 0.2
    elif prediction == 1:
        level = "Low-Moderate"
        score = confidence
    else:
        level = "Low"
        score = 1 - confidence
    
    return RiskAssessment(
        level=level,
        score=min(score, 1.0),
        factors=risk_factors,
        protective_factors=protective_factors
    )

def generate_personalized_recommendations(input_data: MentalHealthInput, prediction: int, confidence: float, risk_assessment: RiskAssessment) -> List[PersonalizedRecommendation]:
    recommendations = []
    
    if prediction == 1:  # Treatment recommended
        if confidence >= 0.8 or risk_assessment.level == "High":
            recommendations.append(PersonalizedRecommendation(
                category="Professional Care",
                priority="Urgent",
                action="Schedule immediate consultation with mental health professional",
                description="Given the assessment results, it's strongly recommended to speak with a licensed therapist, psychologist, or psychiatrist within the next 1-2 weeks.",
                resources=[
                    "National Mental Health Helpline: 1-800-XXX-XXXX",
                    "Psychology Today Therapist Directory",
                    "Your primary care physician for referrals"
                ]
            ))
        
        recommendations.append(PersonalizedRecommendation(
            category="Crisis Support",
            priority="High",
            action="Know your crisis resources",
            description="Always have immediate support available if you experience thoughts of self-harm or crisis situations.",
            resources=[
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "Emergency Services: 911"
            ]
        ))
        
        if input_data.Coping_Struggles == "Yes":
            recommendations.append(PersonalizedRecommendation(
                category="Coping Skills",
                priority="High",
                action="Develop healthy coping mechanisms",
                description="Focus on building stress management and emotional regulation skills through proven techniques.",
                resources=[
                    "Mindfulness meditation apps (Headspace, Calm)",
                    "Cognitive Behavioral Therapy (CBT) techniques",
                    "Local support groups"
                ]
            ))
        
        if input_data.Social_Weakness == "Yes":
            recommendations.append(PersonalizedRecommendation(
                category="Social Support",
                priority="Medium",
                action="Strengthen social connections",
                description="Building and maintaining social relationships is crucial for mental health recovery and maintenance.",
                resources=[
                    "Community social groups",
                    "Volunteer opportunities",
                    "Online support communities"
                ]
            ))
    
    else:  # No treatment needed currently
        recommendations.append(PersonalizedRecommendation(
            category="Prevention",
            priority="Medium",
            action="Maintain mental wellness",
            description="Continue current positive practices and stay vigilant about mental health changes.",
            resources=[
                "Regular self-assessment tools",
                "Mental health education resources",
                "Wellness apps and programs"
            ]
        ))
    
    # Universal recommendations
    recommendations.append(PersonalizedRecommendation(
        category="Lifestyle",
        priority="Medium",
        action="Maintain healthy lifestyle habits",
        description="Regular exercise, adequate sleep, and balanced nutrition significantly impact mental health.",
        resources=[
            "Exercise programs (yoga, walking, sports)",
            "Sleep hygiene resources",
            "Nutrition counseling"
        ]
    ))
    
    return recommendations

def generate_detailed_analysis(input_data: MentalHealthInput, feature_importance: Dict[str, float]) -> DetailedAnalysis:
    primary_concerns = []
    contributing_factors = []
    positive_indicators = []
    areas_of_focus = []
    
    # Analyze based on feature importance and input values
    top_features = list(feature_importance.keys())[:5]
    
    for feature in top_features:
        if feature in ['Growing_Stress', 'Coping_Struggles', 'Mental_Health_History'] and getattr(input_data, feature, 'No') == 'Yes':
            primary_concerns.append(f"{feature.replace('_', ' ').title()}")
    
    # Contributing factors
    if input_data.family_history == "Yes":
        contributing_factors.append("Genetic predisposition and family history")
    if input_data.Days_Indoors in ["31-60 days", "More than 2 months"]:
        contributing_factors.append("Social isolation and reduced activity")
    if input_data.Changes_Habits == "Yes":
        contributing_factors.append("Recent lifestyle disruptions")
    
    # Positive indicators
    if input_data.Work_Interest == "Yes":
        positive_indicators.append("Maintained work engagement and motivation")
    if input_data.mental_health_interview == "Yes":
        positive_indicators.append("Openness to mental health discussions")
    if input_data.care_options == "Yes":
        positive_indicators.append("Awareness of available support systems")
    
    # Areas of focus
    if input_data.Mood_Swings in ["Medium", "High"]:
        areas_of_focus.append("Emotional regulation and mood stability")
    if input_data.Social_Weakness == "Yes":
        areas_of_focus.append("Social skills and relationship building")
    if input_data.Coping_Struggles == "Yes":
        areas_of_focus.append("Stress management and coping strategies")
    
    return DetailedAnalysis(
        primary_concerns=primary_concerns,
        contributing_factors=contributing_factors,
        positive_indicators=positive_indicators,
        areas_of_focus=areas_of_focus
    )

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_data is not None and preprocessor is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=EnhancedPredictionResponse)
async def predict_treatment(
    input_data: MentalHealthInput,
    token: str = Depends(verify_token)
):
    """Predict mental health treatment requirement"""
    
    if model_data is None or preprocessor is None:
        raise HTTPException(
            status_code=500, 
            detail="Model or preprocessor not loaded. Please ensure the model is trained."
        )
    
    try:
        # Preprocess input
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        model = model_data  # model_data is the actual model, not a dictionary
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        
        # Get confidence (probability of predicted class)
        confidence = float(max(probabilities))
        
        # Get feature importance
        feature_names = preprocessor['feature_columns']
        feature_importance = get_feature_importance(
            model, 
            feature_names, 
            processed_data
        )
        
        # Generate enhanced response components
        prediction_label = "Treatment Recommended" if prediction == 1 else "Treatment Not Required"
        
        confidence_level = "Very High" if confidence >= 0.8 else "High" if confidence >= 0.7 else "Moderate" if confidence >= 0.6 else "Low"
        model_certainty = f"The model is {confidence_level.lower()} confident in this prediction based on the provided information."
        prediction_strength = "Strong" if confidence >= 0.75 else "Moderate" if confidence >= 0.6 else "Weak"
        
        confidence_metrics = ConfidenceMetrics(
            overall=confidence,
            level=confidence_level,
            model_certainty=model_certainty,
            prediction_strength=prediction_strength
        )
        
        risk_assessment = generate_risk_assessment(input_data, prediction, confidence, feature_importance)
        detailed_analysis = generate_detailed_analysis(input_data, feature_importance)
        recommendations = generate_personalized_recommendations(input_data, prediction, confidence, risk_assessment)
        
        educational_content = [
            "Understanding Mental Health: Learn about common mental health conditions and their symptoms",
            "The Importance of Early Intervention: How early treatment can improve outcomes",
            "Building Resilience: Strategies for maintaining mental wellness",
            "Support Systems: How family and friends can help in mental health recovery"
        ]
        
        support_resources = {
            "National Suicide Prevention Lifeline": "988",
            "Crisis Text Line": "Text HOME to 741741",
            "NAMI National Helpline": "1-800-950-NAMI (6264)",
            "SAMHSA National Helpline": "1-800-662-4357"
        }
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(input_data.dict())) % 10000}"
        
        return EnhancedPredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            confidence_metrics=confidence_metrics,
            risk_assessment=risk_assessment,
            detailed_analysis=detailed_analysis,
            feature_importance=feature_importance,
            personalized_recommendations=recommendations,
            educational_content=educational_content,
            support_resources=support_resources,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info(token: str = Depends(verify_token)):
    """Get model information"""
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": "RandomForest",
        "accuracy": "Unknown",
        "features_count": len(preprocessor['feature_columns']) if preprocessor else 0,
        "model_type": str(type(model_data).__name__) if model_data else 'Unknown'
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check deployment status"""
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    debug_info = {
        "current_directory": os.getcwd(),
        "script_directory": current_dir,
        "model_loaded": model_data is not None,
        "preprocessor_loaded": preprocessor is not None,
        "directory_contents": os.listdir('.'),
    }
    
    # Check for models directory
    possible_dirs = [
        os.path.join(current_dir, "models"),
        os.path.join(os.path.dirname(current_dir), "models"),
        "models"
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            debug_info[f"models_dir_{dir_path}"] = os.listdir(dir_path)
        else:
            debug_info[f"models_dir_{dir_path}"] = "NOT_FOUND"
    
    return debug_info

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mental Health Treatment Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "debug": "/debug"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 