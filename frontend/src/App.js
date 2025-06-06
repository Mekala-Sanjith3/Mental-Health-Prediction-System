import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';
import PsychologyIcon from '@mui/icons-material/Psychology';

import PredictionForm from './components/PredictionForm';
import ResultsDisplay from './components/ResultsDisplay';
import About from './components/About';
import './App.css';

const theme = createTheme({
  palette: {
    primary: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    secondary: {
      main: '#ff4081',
      light: '#ff79b0',
      dark: '#c60055',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        },
      },
    },
  },
});

function App() {
  const [predictionResult, setPredictionResult] = React.useState(null);

  const handlePredictionResult = (result) => {
    setPredictionResult(result);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <div className="App">
          <AppBar position="static" elevation={2}>
            <Toolbar>
              <PsychologyIcon sx={{ mr: 2 }} />
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Mental Health Treatment Prediction
              </Typography>
            </Toolbar>
          </AppBar>

          <Box sx={{ py: 4 }}>
            <Routes>
              <Route 
                path="/" 
                element={
                  <PredictionForm 
                    onPredictionResult={handlePredictionResult}
                    predictionResult={predictionResult}
                  />
                } 
              />
              <Route 
                path="/results" 
                element={<ResultsDisplay result={predictionResult} />} 
              />
              <Route path="/about" element={<About />} />
            </Routes>
          </Box>

          <Box 
            component="footer" 
            sx={{ 
              py: 3, 
              px: 2, 
              mt: 'auto',
              backgroundColor: 'background.paper',
              borderTop: '1px solid #e0e0e0'
            }}
          >
            <Typography variant="body2" color="text.secondary" align="center">
              © 2025 Mekala Maria Sanjith Reddy.
            </Typography>
          </Box>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App; 