import React from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  Chip,
  Divider,
} from '@mui/material';
import {
  Psychology,
  DataUsage,
  Security,
  Code,
  Warning,
  CheckCircle,
  School,
} from '@mui/icons-material';

const About = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center" color="primary">
          About Mental Health Prediction System
        </Typography>
        
        <Typography variant="h6" align="center" color="text.secondary" paragraph>
          An AI-powered tool for mental health treatment prediction using machine learning
        </Typography>

        <Grid container spacing={4} sx={{ mt: 2 }}>
          {/* Project Overview */}
          <Grid item xs={12}>
            <Card elevation={3}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <Psychology color="primary" />
                  <Typography variant="h5" component="h2" sx={{ ml: 1 }}>
                    Project Overview
                  </Typography>
                </Box>
                
                <Typography variant="body1" paragraph>
                  This system uses advanced machine learning algorithms to analyze mental health survey data 
                  and predict whether an individual might benefit from mental health treatment. The prediction 
                  is based on various factors including personal history, current symptoms, and lifestyle indicators.
                </Typography>

                <Typography variant="body1" paragraph>
                  The goal is to provide an accessible, preliminary assessment tool that can help individuals 
                  understand their mental health status and encourage them to seek professional help when needed.
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Technology Stack */}
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <Code color="primary" />
                  <Typography variant="h5" component="h2" sx={{ ml: 1 }}>
                    Technology Stack
                  </Typography>
                </Box>

                <Typography variant="h6" gutterBottom>Backend:</Typography>
                <Box sx={{ mb: 2 }}>
                  <Chip label="FastAPI" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="Python" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="Scikit-learn" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="XGBoost" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="Pandas" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="NumPy" sx={{ mr: 1, mb: 1 }} />
                </Box>

                <Typography variant="h6" gutterBottom>Frontend:</Typography>
                <Box sx={{ mb: 2 }}>
                  <Chip label="React" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="Material-UI" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="Recharts" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="Axios" sx={{ mr: 1, mb: 1 }} />
                </Box>

                <Typography variant="h6" gutterBottom>Deployment:</Typography>
                <Box>
                  <Chip label="Docker" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="Docker Compose" sx={{ mr: 1, mb: 1 }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Machine Learning Models */}
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <School color="primary" />
                  <Typography variant="h5" component="h2" sx={{ ml: 1 }}>
                    ML Models
                  </Typography>
                </Box>

                <Typography variant="body1" paragraph>
                  The system implements and compares multiple machine learning algorithms:
                </Typography>

                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Random Forest"
                      secondary="Ensemble method with feature importance"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="XGBoost"
                      secondary="Gradient boosting for high performance"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Logistic Regression"
                      secondary="Linear model for interpretability"
                    />
                  </ListItem>
                </List>

                <Typography variant="body2" color="text.secondary">
                  The best performing model is automatically selected based on F1-score evaluation.
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Data Processing */}
          <Grid item xs={12}>
            <Card elevation={3}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <DataUsage color="primary" />
                  <Typography variant="h5" component="h2" sx={{ ml: 1 }}>
                    Data Processing & Features
                  </Typography>
                </Box>

                <Typography variant="body1" paragraph>
                  The system processes various types of mental health indicators:
                </Typography>

                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Typography variant="h6" gutterBottom>Personal Information</Typography>
                    <List dense>
                      <ListItem><ListItemText primary="Gender" /></ListItem>
                      <ListItem><ListItemText primary="Country" /></ListItem>
                      <ListItem><ListItemText primary="Occupation" /></ListItem>
                      <ListItem><ListItemText primary="Employment Status" /></ListItem>
                    </List>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Typography variant="h6" gutterBottom>Mental Health History</Typography>
                    <List dense>
                      <ListItem><ListItemText primary="Family History" /></ListItem>
                      <ListItem><ListItemText primary="Personal History" /></ListItem>
                      <ListItem><ListItemText primary="Previous Interviews" /></ListItem>
                      <ListItem><ListItemText primary="Care Options Awareness" /></ListItem>
                    </List>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Typography variant="h6" gutterBottom>Current Status</Typography>
                    <List dense>
                      <ListItem><ListItemText primary="Days Spent Indoors" /></ListItem>
                      <ListItem><ListItemText primary="Growing Stress" /></ListItem>
                      <ListItem><ListItemText primary="Mood Swings" /></ListItem>
                      <ListItem><ListItemText primary="Work Interest" /></ListItem>
                    </List>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Security & Privacy */}
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <Security color="primary" />
                  <Typography variant="h5" component="h2" sx={{ ml: 1 }}>
                    Security & Privacy
                  </Typography>
                </Box>

                <List>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Token-based Authentication"
                      secondary="Secure API access with bearer tokens"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="No Data Storage"
                      secondary="Predictions are not stored permanently"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="HTTPS Encryption"
                      secondary="All data transmission is encrypted"
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Model Performance */}
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <DataUsage color="primary" />
                  <Typography variant="h5" component="h2" sx={{ ml: 1 }}>
                    Model Performance
                  </Typography>
                </Box>

                <Typography variant="body1" paragraph>
                  Our models are evaluated using multiple metrics:
                </Typography>

                <List>
                  <ListItem>
                    <ListItemText 
                      primary="Accuracy"
                      secondary="Overall prediction correctness"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="Precision & Recall"
                      secondary="Balance between false positives and negatives"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="F1-Score"
                      secondary="Harmonic mean of precision and recall"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="ROC-AUC"
                      secondary="Area under the receiver operating curve"
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Important Disclaimers */}
          <Grid item xs={12}>
            <Alert severity="warning" sx={{ mb: 2 }}>
              <Typography variant="h6" gutterBottom>
                <Warning sx={{ mr: 1, verticalAlign: 'middle' }} />
                Important Medical Disclaimer
              </Typography>
              <Typography variant="body1" paragraph>
                This tool is for educational and informational purposes only. It is NOT a substitute for 
                professional medical advice, diagnosis, or treatment. The predictions made by this system 
                should not be used as the sole basis for making healthcare decisions.
              </Typography>
              <Typography variant="body1">
                If you are experiencing mental health concerns, please consult with a qualified healthcare 
                professional, licensed therapist, or mental health specialist. In case of emergency or 
                suicidal thoughts, please contact your local emergency services or a crisis helpline immediately.
              </Typography>
            </Alert>
          </Grid>

          {/* Contact & Support */}
          <Grid item xs={12}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h5" component="h2" gutterBottom>
                  Contact & Support
                </Typography>
                
                <Typography variant="body1" paragraph>
                  This project was developed as a demonstration of machine learning applications in healthcare. 
                  For technical questions or support, please refer to the project documentation.
                </Typography>

                <Divider sx={{ my: 2 }} />

                <Typography variant="body2" color="text.secondary">
                  © 2025 Mental Health Prediction System. - By Sanjith Mekala
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default About; 