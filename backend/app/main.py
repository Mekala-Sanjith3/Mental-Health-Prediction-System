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
from .fallback_model import FallbackModel, create_fallback_preprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Treatment Prediction API",
    description="API for predicting mental health treatment requirements using machine learning",
    version="1.0.0"
)

# CORS middleware - Updated for Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://*.vercel.app",  # Allow Vercel domains
        "*"  # For development - remove in production
    ],
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

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: int = Field(..., description="Prediction result")
    prediction_label: str = Field(..., description="Human readable prediction")
    confidence: float = Field(..., description="Prediction confidence")
    feature_importance: Dict[str, float] = Field(..., description="Feature importances")
    timestamp: str = Field(..., description="Prediction timestamp")

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
        # Try different paths for Vercel deployment
        model_paths = [
            "../models/best_model.pkl",
            "./models/best_model.pkl",
            "models/best_model.pkl",
            "backend/models/best_model.pkl"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                logger.info(f"Model loaded from {model_path}: {model_data.get('model_name', 'Unknown')}")
                break
        else:
            logger.warning("Model file not found in any expected location")
            
        # Try different paths for preprocessor
        preprocessor_paths = [
            "../models/preprocessor.pkl",
            "./models/preprocessor.pkl", 
            "models/preprocessor.pkl",
            "backend/models/preprocessor.pkl"
        ]
        
        for preprocessor_path in preprocessor_paths:
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, 'rb') as f:
                    preprocessor = pickle.load(f)
                logger.info(f"Preprocessor loaded from {preprocessor_path}")
                break
        else:
            logger.warning("Preprocessor file not found in any expected location")
            
    except Exception as e:
        logger.error(f"Error loading model/preprocessor: {e}")
        
    # If model or preprocessor not loaded, use fallback
    if model_data is None:
        logger.info("Using fallback model")
        model_data = {
            'model': FallbackModel(),
            'model_name': 'Fallback Rule-Based Model',
            'accuracy': 0.65  # Estimated
        }
        
    if preprocessor is None:
        logger.info("Using fallback preprocessor")
        preprocessor = create_fallback_preprocessor()

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
        # For linear models
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return {}
        
        # Get top 5 features
        top_indices = np.argsort(importance)[-5:]
        top_features = {
            feature_names[i]: float(importance[i]) 
            for i in top_indices
        }
        
        return top_features
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return {}

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

@app.post("/predict", response_model=PredictionResponse)
async def predict_treatment(
    input_data: MentalHealthInput,
    token: str = Depends(verify_token)
):
    """Predict mental health treatment requirement"""
    
    if model_data is None or preprocessor is None:
        raise HTTPException(
            status_code=500, 
            detail="Model or preprocessor not loaded"
        )
    
    try:
        # Preprocess input
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        model = model_data['model']
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        # Get confidence (probability of predicted class)
        confidence = float(max(prediction_proba))
        
        # Get feature importance
        feature_importance = get_feature_importance(
            model, 
            model_data['feature_names'], 
            processed_data
        )
        
        # Create response
        prediction_label = "Treatment Needed" if prediction == 1 else "No Treatment Needed"
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            confidence=confidence,
            feature_importance=feature_importance,
            timestamp=datetime.now().isoformat()
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
        "model_name": model_data.get("model_name", "Unknown"),
        "model_type": str(type(model_data["model"]).__name__),
        "feature_count": len(model_data.get("feature_names", [])),
        "status": "loaded"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mental Health Treatment Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 