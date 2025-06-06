import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Grid,
  Alert,
  List,
  ListItem,
  ListItemText,
  Paper,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Container,
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  TrendingUp,
  ExpandMore,
  Info,
  SelfImprovement,
  Support,
  MenuBook,
  TrendingDown,
  Assessment,
  Analytics,
  Shield,
  PriorityHigh,
  PhoneInTalk,
  FavoriteOutlined,
  Timeline,
  PersonPin,
} from '@mui/icons-material';
import { 
  PieChart, 
  Pie, 
  Cell, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer
} from 'recharts';

const ResultsDisplay = ({ result, onReset }) => {
  const [expandedSections, setExpandedSections] = useState({
    analysis: true,
    recommendations: true,
    resources: false
  });

  if (!result) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="info" sx={{ textAlign: 'center' }}>
          No prediction results to display. Please submit the form to get a comprehensive mental health assessment.
        </Alert>
      </Container>
    );
  }

  const {
    prediction,
    prediction_label,
    confidence_metrics,
    risk_assessment,
    detailed_analysis,
    feature_importance,
    personalized_recommendations,
    educational_content,
    support_resources,
    timestamp,
    session_id
  } = result;

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // Enhanced data preparation
  const confidenceData = [
    { 
      name: 'Confidence', 
      value: confidence_metrics.overall * 100,
      color: confidence_metrics.overall >= 0.8 ? '#4caf50' : confidence_metrics.overall >= 0.6 ? '#ff9800' : '#f44336'
    },
    { 
      name: 'Uncertainty', 
      value: (1 - confidence_metrics.overall) * 100,
      color: '#e0e0e0'
    }
  ];

  const riskData = [
    { 
      name: 'Risk Score', 
      value: risk_assessment.score * 100,
      color: risk_assessment.level === 'High' ? '#f44336' : 
             risk_assessment.level === 'Moderate' ? '#ff9800' : '#4caf50'
    },
    { 
      name: 'Remaining', 
      value: (1 - risk_assessment.score) * 100,
      color: '#e0e0e0'
    }
  ];

  const featureData = Object.entries(feature_importance || {}).map(([feature, importance]) => ({
    feature: feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    importance: importance,
    percentage: (importance * 100).toFixed(1)
  })).sort((a, b) => b.importance - a.importance).slice(0, 8);

  const COLORS = ['#2196f3', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#00bcd4', '#8bc34a', '#ffc107'];
  const getResultIcon = () => prediction === 1 ? <Warning /> : <CheckCircle />;
  
  const getRiskLevelColor = () => {
    switch (risk_assessment.level) {
      case 'High': return 'error';
      case 'Moderate': return 'warning';
      case 'Low-Moderate': return 'info';
      default: return 'success';
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority.toLowerCase()) {
      case 'urgent': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      default: return 'default';
    }
  };

  const getPriorityIcon = (priority) => {
    switch (priority.toLowerCase()) {
      case 'urgent': return <PriorityHigh />;
      case 'high': return <Warning />;
      case 'medium': return <Info />;
      default: return <Info />;
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header Section */}
      <Card elevation={3} sx={{ mb: 4, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <CardContent sx={{ textAlign: 'center', py: 4 }}>
          <Box display="flex" justifyContent="center" alignItems="center" mb={2}>
            {getResultIcon()}
            <Typography variant="h3" component="h1" sx={{ ml: 2, fontWeight: 'bold' }}>
              Assessment Complete
            </Typography>
          </Box>
          <Typography variant="h5" sx={{ mb: 2, opacity: 0.9 }}>
            {prediction_label}
          </Typography>
          <Box display="flex" justifyContent="center" alignItems="center" gap={3}>
            <Chip 
              label={`Confidence: ${confidence_metrics.level}`} 
              color={confidence_metrics.overall >= 0.7 ? 'success' : 'warning'}
              variant="filled"
              sx={{ color: 'white', fontWeight: 'bold' }}
            />
            <Chip 
              label={`Risk Level: ${risk_assessment.level}`} 
              color={getRiskLevelColor()}
              variant="filled"
              sx={{ color: 'white', fontWeight: 'bold' }}
            />
            <Chip 
              label={`Session: ${session_id}`} 
              variant="outlined"
              sx={{ color: 'white', borderColor: 'white' }}
            />
          </Box>
        </CardContent>
      </Card>

      <Grid container spacing={4}>
        {/* Confidence and Risk Metrics */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                <Analytics sx={{ mr: 1, verticalAlign: 'middle' }} />
                Confidence Analysis
              </Typography>
              <Box sx={{ height: 200, mb: 2 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={confidenceData}
                      cx="50%"
                      cy="50%"
                      outerRadius={60}
                      fill="#8884d8"
                      dataKey="value"
                      label={({value}) => `${value.toFixed(1)}%`}
                    >
                      {confidenceData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
              <Typography variant="body2" color="textSecondary">
                {confidence_metrics.model_certainty}
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                <strong>Prediction Strength:</strong> {confidence_metrics.prediction_strength}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                <Shield sx={{ mr: 1, verticalAlign: 'middle' }} />
                Risk Assessment
              </Typography>
              <Box sx={{ height: 200, mb: 2 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={riskData}
                      cx="50%"
                      cy="50%"
                      outerRadius={60}
                      fill="#8884d8"
                      dataKey="value"
                      label={({value}) => `${value.toFixed(1)}%`}
                    >
                      {riskData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
              <Typography variant="body2">
                <strong>Risk Factors:</strong> {risk_assessment.factors.length}
              </Typography>
              <Typography variant="body2">
                <strong>Protective Factors:</strong> {risk_assessment.protective_factors.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Feature Importance */}
        <Grid item xs={12}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
                Key Assessment Factors
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={featureData} margin={{ top: 20, right: 30, left: 40, bottom: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="feature" 
                      angle={-45}
                      textAnchor="end"
                      height={80}
                      fontSize={12}
                    />
                    <YAxis />
                    <RechartsTooltip 
                      formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Importance']}
                    />
                    <Bar dataKey="importance" fill="#2196f3" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Detailed Analysis */}
        <Grid item xs={12}>
          <Accordion expanded={expandedSections.analysis} onChange={() => toggleSection('analysis')}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6" color="primary">
                <PersonPin sx={{ mr: 1, verticalAlign: 'middle' }} />
                Detailed Analysis
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                {detailed_analysis.primary_concerns.length > 0 && (
                  <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2, bgcolor: '#ffebee' }}>
                      <Typography variant="h6" color="error" gutterBottom>
                        <Warning sx={{ mr: 1, verticalAlign: 'middle' }} />
                        Primary Concerns
                      </Typography>
                      <List dense>
                        {detailed_analysis.primary_concerns.map((concern, index) => (
                          <ListItem key={index}>
                            <ListItemText primary={concern} />
                          </ListItem>
                        ))}
                      </List>
                    </Paper>
                  </Grid>
                )}

                {detailed_analysis.contributing_factors.length > 0 && (
                  <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2, bgcolor: '#fff3e0' }}>
                      <Typography variant="h6" color="warning.main" gutterBottom>
                        <TrendingDown sx={{ mr: 1, verticalAlign: 'middle' }} />
                        Contributing Factors
                      </Typography>
                      <List dense>
                        {detailed_analysis.contributing_factors.map((factor, index) => (
                          <ListItem key={index}>
                            <ListItemText primary={factor} />
                          </ListItem>
                        ))}
                      </List>
                    </Paper>
                  </Grid>
                )}

                {detailed_analysis.positive_indicators.length > 0 && (
                  <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2, bgcolor: '#e8f5e8' }}>
                      <Typography variant="h6" color="success.main" gutterBottom>
                        <FavoriteOutlined sx={{ mr: 1, verticalAlign: 'middle' }} />
                        Positive Indicators
                      </Typography>
                      <List dense>
                        {detailed_analysis.positive_indicators.map((indicator, index) => (
                          <ListItem key={index}>
                            <ListItemText primary={indicator} />
                          </ListItem>
                        ))}
                      </List>
                    </Paper>
                  </Grid>
                )}

                {detailed_analysis.areas_of_focus.length > 0 && (
                  <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2, bgcolor: '#e3f2fd' }}>
                      <Typography variant="h6" color="info.main" gutterBottom>
                        <Timeline sx={{ mr: 1, verticalAlign: 'middle' }} />
                        Areas of Focus
                      </Typography>
                      <List dense>
                        {detailed_analysis.areas_of_focus.map((area, index) => (
                          <ListItem key={index}>
                            <ListItemText primary={area} />
                          </ListItem>
                        ))}
                      </List>
                    </Paper>
                  </Grid>
                )}
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Personalized Recommendations */}
        <Grid item xs={12}>
          <Accordion expanded={expandedSections.recommendations} onChange={() => toggleSection('recommendations')}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6" color="primary">
                <SelfImprovement sx={{ mr: 1, verticalAlign: 'middle' }} />
                Personalized Recommendations ({personalized_recommendations.length})
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {personalized_recommendations.map((rec, index) => (
                  <Grid item xs={12} md={6} key={index}>
                    <Card variant="outlined" sx={{ height: '100%' }}>
                      <CardContent>
                        <Box display="flex" alignItems="center" mb={2}>
                          {getPriorityIcon(rec.priority)}
                          <Typography variant="h6" sx={{ ml: 1, flexGrow: 1 }}>
                            {rec.action}
                          </Typography>
                          <Chip 
                            label={rec.priority} 
                            color={getPriorityColor(rec.priority)}
                            size="small"
                          />
                        </Box>
                        <Typography variant="body2" color="textSecondary" gutterBottom>
                          Category: {rec.category}
                        </Typography>
                        <Typography variant="body2" paragraph>
                          {rec.description}
                        </Typography>
                        {rec.resources.length > 0 && (
                          <>
                            <Typography variant="subtitle2" gutterBottom>
                              Resources:
                            </Typography>
                            <List dense>
                              {rec.resources.map((resource, idx) => (
                                <ListItem key={idx} sx={{ py: 0 }}>
                                  <ListItemText 
                                    primary={resource}
                                    primaryTypographyProps={{ variant: 'body2' }}
                                  />
                                </ListItem>
                              ))}
                            </List>
                          </>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Support Resources and Educational Content */}
        <Grid item xs={12}>
          <Accordion expanded={expandedSections.resources} onChange={() => toggleSection('resources')}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6" color="primary">
                <Support sx={{ mr: 1, verticalAlign: 'middle' }} />
                Support Resources & Education
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                {/* Crisis Support */}
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 3, bgcolor: '#fff3e0', border: '2px solid #ff9800' }}>
                    <Typography variant="h6" color="warning.main" gutterBottom>
                      <PhoneInTalk sx={{ mr: 1, verticalAlign: 'middle' }} />
                      Crisis Support
                    </Typography>
                    <Typography variant="body2" gutterBottom sx={{ fontWeight: 'bold' }}>
                      If you're in crisis, reach out immediately:
                    </Typography>
                    {Object.entries(support_resources).map(([name, contact], index) => (
                      <Box key={index} sx={{ mb: 1 }}>
                        <Typography variant="body2">
                          <strong>{name}:</strong> {contact}
                        </Typography>
                      </Box>
                    ))}
                  </Paper>
                </Grid>

                {/* Educational Content */}
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 3, bgcolor: '#e8f5e8' }}>
                    <Typography variant="h6" color="success.main" gutterBottom>
                      <MenuBook sx={{ mr: 1, verticalAlign: 'middle' }} />
                      Educational Resources
                    </Typography>
                    <List dense>
                      {educational_content.map((content, index) => (
                        <ListItem key={index}>
                          <ListItemText primary={content} />
                        </ListItem>
                      ))}
                    </List>
                  </Paper>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Assessment Summary */}
        <Grid item xs={12}>
          <Card elevation={1} sx={{ bgcolor: '#f5f5f5' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
                Assessment Summary
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    Assessment Date
                  </Typography>
                  <Typography variant="body1">
                    {new Date(timestamp).toLocaleDateString()}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    Session ID
                  </Typography>
                  <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
                    {session_id}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    Risk Factors Identified
                  </Typography>
                  <Typography variant="body1">
                    {risk_assessment.factors.length}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    Recommendations Given
                  </Typography>
                  <Typography variant="body1">
                    {personalized_recommendations.length}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Action Buttons */}
        <Grid item xs={12} sx={{ textAlign: 'center', mt: 2 }}>
          <Button 
            variant="contained" 
            color="primary" 
            size="large"
            onClick={onReset}
            sx={{ mx: 1 }}
          >
            Take New Assessment
          </Button>
          <Button 
            variant="outlined" 
            color="primary" 
            size="large"
            onClick={() => window.print()}
            sx={{ mx: 1 }}
          >
            Print Results
          </Button>
        </Grid>
      </Grid>

      {/* Disclaimer */}
      <Alert severity="warning" sx={{ mt: 4 }}>
        <Typography variant="body2">
          <strong>Important:</strong> This assessment is for educational purposes only and should not replace professional medical advice. 
          If you're experiencing mental health concerns, please consult with a qualified mental health professional.
        </Typography>
      </Alert>
    </Container>
  );
};

export default ResultsDisplay; 