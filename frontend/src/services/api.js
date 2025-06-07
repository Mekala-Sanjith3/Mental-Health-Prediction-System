import axios from 'axios';

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status;
    const errorMap = {
      401: 'Invalid authentication token',
      403: 'Access forbidden. Authentication required',
      422: 'Validation error. Please check your input',
      500: 'Server error. Please try again later',
      503: 'Service unavailable. Please try again later'
    };
    
    const message = errorMap[status] || error.response?.data?.detail || error.message || 'An error occurred';
    throw new Error(message);
  }
);

const PredictionAPI = {
  async healthCheck() {
    const response = await apiClient.get('/health');
    return response.data;
  },

  async predict(data, token = 'demo-token-12345') {
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.post('/predict', data, { headers });
    
    const result = response.data;
    
    return {
      ...result,
      confidence: result.confidence_metrics?.overall || 0,
      prediction: result.prediction,
      prediction_label: result.prediction_label,
      feature_importance: result.feature_importance || {},
      timestamp: result.timestamp,
      
      confidence_metrics: result.confidence_metrics || {
        overall: result.confidence || 0,
        level: 'Unknown',
        model_certainty: 'Confidence information not available',
        prediction_strength: 'Unknown'
      },
      risk_assessment: result.risk_assessment || {
        level: 'Unknown',
        score: 0,
        factors: [],
        protective_factors: []
      },
      detailed_analysis: result.detailed_analysis || {
        primary_concerns: [],
        contributing_factors: [],
        positive_indicators: [],
        areas_of_focus: []
      },
      personalized_recommendations: result.personalized_recommendations || [],
      educational_content: result.educational_content || [],
      support_resources: result.support_resources || {},
      session_id: result.session_id || 'unknown_session'
    };
  },

  async getModelInfo(token = 'demo-token-12345') {
    try {
      const headers = token ? { Authorization: `Bearer ${token}` } : {};
      const response = await apiClient.get('/model-info', { headers });
      return response.data;
    } catch (error) {
      return {
        model_name: 'RandomForest',
        model_type: 'RandomForestClassifier',
        status: 'loaded',
        accuracy: 'Unknown',
        features_count: 0
      };
    }
  },

  async testConnection() {
    const response = await apiClient.get('/');
    return response.data;
  },


};

export default PredictionAPI; 