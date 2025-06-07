import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
} from '@mui/material';
import { ExpandMore, BugReport, CheckCircle, Error } from '@mui/icons-material';
import PredictionAPI from '../services/api';

const DebugPanel = () => {
  const [loading, setLoading] = useState(false);
  const [debugInfo, setDebugInfo] = useState(null);
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState(null);

  const testConnection = async () => {
    setLoading(true);
    setError(null);
    setConnectionStatus(null);
    
    try {
      // Test basic connection
      const rootResponse = await PredictionAPI.testConnection();
      setConnectionStatus({ type: 'success', message: 'API connection successful', data: rootResponse });
      
      // Test health endpoint
      const healthResponse = await PredictionAPI.healthCheck();
      
      // Test debug endpoint
      const debugResponse = await PredictionAPI.getDebugInfo();
      
      setDebugInfo({
        root: rootResponse,
        health: healthResponse,
        debug: debugResponse
      });
      
    } catch (err) {
      setError(err.message);
      setConnectionStatus({ type: 'error', message: `Connection failed: ${err.message}` });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card sx={{ mb: 4 }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <BugReport sx={{ mr: 1 }} />
          <Typography variant="h6">API Debug Panel</Typography>
        </Box>
        
        <Button
          variant="contained"
          onClick={testConnection}
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : <BugReport />}
          sx={{ mb: 2 }}
        >
          {loading ? 'Testing...' : 'Test API Connection'}
        </Button>

        {connectionStatus && (
          <Alert 
            severity={connectionStatus.type} 
            sx={{ mb: 2 }}
            icon={connectionStatus.type === 'success' ? <CheckCircle /> : <Error />}
          >
            {connectionStatus.message}
          </Alert>
        )}

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {debugInfo && (
          <Box>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="subtitle1">API Root Response</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <pre style={{ fontSize: '12px', overflow: 'auto' }}>
                  {JSON.stringify(debugInfo.root, null, 2)}
                </pre>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography variant="subtitle1">Health Check</Typography>
                  <Chip 
                    label={debugInfo.health?.model_loaded ? 'Model Loaded' : 'Model Not Loaded'} 
                    color={debugInfo.health?.model_loaded ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <pre style={{ fontSize: '12px', overflow: 'auto' }}>
                  {JSON.stringify(debugInfo.health, null, 2)}
                </pre>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="subtitle1">Debug Information</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <pre style={{ fontSize: '12px', overflow: 'auto' }}>
                  {JSON.stringify(debugInfo.debug, null, 2)}
                </pre>
              </AccordionDetails>
            </Accordion>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default DebugPanel; 