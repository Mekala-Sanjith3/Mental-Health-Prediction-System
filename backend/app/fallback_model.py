import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class FallbackModel:
    """Simple fallback model for when trained model is not available"""
    
    def __init__(self):
        self.model_name = "Simple Rule-Based Fallback"
        self.version = "1.0"
        
    def predict_proba(self, X):
        """Simple rule-based prediction with probabilities"""
        predictions = []
        
        for row in X:
            # Simple scoring based on basic rules
            score = 0
            
            # Add points for various risk factors (simplified)
            # Note: This is a very basic fallback - real model should be trained
            
            # Family history (assumed to be at index 3)
            if len(row) > 3 and row[3] == 1:
                score += 0.3
                
            # Mental health history (assumed to be at index 8)
            if len(row) > 8 and row[8] == 1:
                score += 0.4
                
            # Coping struggles (assumed to be at index 10)
            if len(row) > 10 and row[10] == 1:
                score += 0.2
                
            # Social weakness (assumed to be at index 12)
            if len(row) > 12 and row[12] == 1:
                score += 0.1
            
            # Normalize score to probability
            prob_treatment = min(max(score, 0.1), 0.9)  # Keep between 0.1 and 0.9
            prob_no_treatment = 1 - prob_treatment
            
            predictions.append([prob_no_treatment, prob_treatment])
            
        return np.array(predictions)
    
    def predict(self, X):
        """Simple rule-based prediction"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
        
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Return mock feature importance"""
        n_features = len(feature_names)
        importance = np.random.random(n_features)
        importance = importance / importance.sum()  # Normalize
        
        return dict(zip(feature_names, importance.tolist()))

def create_fallback_preprocessor():
    """Create a basic preprocessor structure"""
    return {
        'scaler': MockScaler(),
        'label_encoders': {},  # Empty for fallback
        'feature_columns': [
            'Gender', 'Country', 'Occupation', 'self_employed', 'family_history',
            'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History',
            'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness',
            'mental_health_interview', 'care_options'
        ]
    }

class MockScaler:
    """Mock scaler that doesn't actually scale"""
    
    def transform(self, X):
        """Return data as-is (no scaling)"""
        return np.array(X)
        
    def fit_transform(self, X):
        """Return data as-is (no scaling)"""
        return np.array(X) 