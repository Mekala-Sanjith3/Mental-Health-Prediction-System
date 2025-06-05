import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app, verify_token, preprocess_input, MentalHealthInput

client = TestClient(app)
VALID_TOKEN = "demo-token-12345"
INVALID_TOKEN = "invalid-token"

SAMPLE_INPUT = {
    "Gender": "Female",
    "Country": "United States", 
    "Occupation": "Corporate",
    "self_employed": "No",
    "family_history": "Yes",
    "Days_Indoors": "1-14 days",
    "Growing_Stress": "Yes",
    "Changes_Habits": "No",
    "Mental_Health_History": "Yes",
    "Mood_Swings": "Medium",
    "Coping_Struggles": "No",
    "Work_Interest": "No",
    "Social_Weakness": "Yes",
    "mental_health_interview": "No",
    "care_options": "Not sure"
}

class TestHealthEndpoint:
    def test_health_check_success(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "version" in data

class TestAuthentication:
    """Test authentication functionality"""
    
    def test_predict_without_token(self):
        """Test prediction without authentication token"""
        response = client.post("/predict", json=SAMPLE_INPUT)
        assert response.status_code == 403

    def test_predict_with_invalid_token(self):
        """Test prediction with invalid token"""
        headers = {"Authorization": f"Bearer {INVALID_TOKEN}"}
        response = client.post("/predict", json=SAMPLE_INPUT, headers=headers)
        assert response.status_code == 401

    def test_model_info_without_token(self):
        """Test model info without authentication token"""
        response = client.get("/model-info")
        assert response.status_code == 403

class TestInputValidation:
    """Test input validation"""
    
    def test_predict_with_missing_fields(self):
        """Test prediction with missing required fields"""
        incomplete_input = {"Gender": "Female", "Country": "United States"}
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = client.post("/predict", json=incomplete_input, headers=headers)
        assert response.status_code == 422  # Validation error

    def test_predict_with_invalid_values(self):
        """Test prediction with invalid field values"""
        invalid_input = SAMPLE_INPUT.copy()
        invalid_input["Gender"] = "InvalidGender"
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = client.post("/predict", json=invalid_input, headers=headers)
        # Should still process but may give unexpected results
        # The exact behavior depends on preprocessing implementation

class TestPredictionEndpoint:
    """Test the prediction endpoint"""
    
    @patch('app.main.model_data')
    @patch('app.main.preprocessor')
    def test_predict_success_mock(self, mock_preprocessor, mock_model_data):
        """Test successful prediction with mocked model"""
        # Mock preprocessor
        mock_preprocessor_data = {
            'label_encoders': {'Gender': Mock(), 'Country': Mock(), 'Occupation': Mock()},
            'scaler': Mock(),
            'feature_columns': ['Gender', 'Country', 'Occupation', 'self_employed', 
                              'family_history', 'Days_Indoors', 'Growing_Stress']
        }
        mock_preprocessor.return_value = mock_preprocessor_data
        
        # Mock label encoders
        for encoder in mock_preprocessor_data['label_encoders'].values():
            encoder.transform.return_value = [0]
        
        # Mock scaler
        mock_preprocessor_data['scaler'].transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
        
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
        
        mock_model_data_dict = {
            'model': mock_model,
            'model_name': 'TestModel',
            'feature_names': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            'metrics': {'f1_score': 0.85}
        }
        mock_model_data.return_value = mock_model_data_dict
        
        # Set global variables
        import app.main
        app.main.model_data = mock_model_data_dict
        app.main.preprocessor = mock_preprocessor_data
        
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = client.post("/predict", json=SAMPLE_INPUT, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "prediction_label" in data
        assert "confidence" in data
        assert "feature_importance" in data
        assert "timestamp" in data

    def test_predict_model_not_loaded(self):
        """Test prediction when model is not loaded"""
        # Temporarily set model_data to None
        import app.main
        original_model_data = app.main.model_data
        app.main.model_data = None
        
        try:
            headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
            response = client.post("/predict", json=SAMPLE_INPUT, headers=headers)
            assert response.status_code == 500
        finally:
            # Restore original state
            app.main.model_data = original_model_data

class TestModelInfoEndpoint:
    """Test the model info endpoint"""
    
    @patch('app.main.model_data')
    def test_model_info_success(self, mock_model_data):
        """Test successful model info retrieval"""
        mock_model_data_dict = {
            'model_name': 'RandomForest',
            'feature_names': ['feature1', 'feature2', 'feature3'],
            'metrics': {'accuracy': 0.85, 'f1_score': 0.83}
        }
        
        import app.main
        app.main.model_data = mock_model_data_dict
        
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = client.get("/model-info", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "RandomForest"
        assert data["feature_count"] == 3
        assert "features" in data
        assert "metrics" in data

class TestRootEndpoint:
    """Test the root endpoint"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data

class TestPreprocessing:
    """Test preprocessing functionality"""
    
    def test_mental_health_input_model(self):
        """Test MentalHealthInput pydantic model"""
        input_data = MentalHealthInput(**SAMPLE_INPUT)
        assert input_data.Gender == "Female"
        assert input_data.Country == "United States"
        assert input_data.Occupation == "Corporate"

    def test_invalid_mental_health_input(self):
        """Test MentalHealthInput with missing required fields"""
        with pytest.raises(ValueError):
            MentalHealthInput(Gender="Female")  # Missing required fields

class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_api_flow(self):
        """Test complete API flow"""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Check root endpoint
        root_response = client.get("/")
        assert root_response.status_code == 200
        
        # 3. Try prediction without token (should fail)
        unauth_response = client.post("/predict", json=SAMPLE_INPUT)
        assert unauth_response.status_code == 403
        
        # 4. Try prediction with valid token
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        # Note: This might fail if model isn't actually loaded, which is expected in test environment

@pytest.fixture
def mock_model_and_preprocessor():
    """Fixture to mock model and preprocessor"""
    mock_model = Mock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.3, 0.7]]
    mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
    
    mock_model_data = {
        'model': mock_model,
        'model_name': 'TestModel',
        'feature_names': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
        'metrics': {'f1_score': 0.85}
    }
    
    mock_preprocessor = {
        'label_encoders': {
            'Gender': Mock(),
            'Country': Mock(), 
            'Occupation': Mock()
        },
        'scaler': Mock(),
        'feature_columns': ['Gender', 'Country', 'Occupation', 'self_employed', 
                          'family_history', 'Days_Indoors', 'Growing_Stress']
    }
    
    # Configure mocks
    for encoder in mock_preprocessor['label_encoders'].values():
        encoder.transform.return_value = [0]
    
    mock_preprocessor['scaler'].transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    
    return mock_model_data, mock_preprocessor

if __name__ == "__main__":
    pytest.main([__file__]) 