import axios from 'axios';

// For Vercel deployment, use relative URLs in production
const BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api'  // Vercel will route /api/* to backend
  : (process.env.REACT_APP_API_URL || 'http://localhost:8000');

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
    return response.data;
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
        status: 'loaded'
      };
    }
  },

  async testConnection() {
    const response = await apiClient.get('/');
    return response.data;
  },
};

export default PredictionAPI; 