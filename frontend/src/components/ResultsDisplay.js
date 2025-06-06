import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Grid,
  Alert,
  List,
  ListItem,
  ListItemText,
  Paper,
  Button,
  Collapse,
  IconButton,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Avatar,
  Stack,
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  TrendingUp,
  AccessTime,
  ExpandMore,
  Info,
  LocalHospital,
  Phone,
  School,
  LightbulbOutlined,
  HealthAndSafety,
  SelfImprovement,
  Support,
  MenuBook,
  Assessment,
  Analytics,
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
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';

const ResultsDisplay = ({ result, onReset }) => {
  const [showDetailedAnalysis, setShowDetailedAnalysis] = useState(false);

  if (!result) {
    return (
      <Alert severity="info">
        No prediction results to display. Please submit the form to get a prediction.
      </Alert>
    );
  }

  const {
    prediction,
    prediction_label,
    confidence,
    feature_importance,
    timestamp
  } = result;



  // Enhanced data preparation
  const pieData = [
    { 
      name: 'Treatment Recommended', 
      value: prediction === 1 ? confidence : 1 - confidence,
      color: '#ff6b6b',
      description: prediction === 1 ? 'Model suggests treatment may be beneficial' : 'Lower probability of needing treatment'
    },
    { 
      name: 'No Treatment Needed', 
      value: prediction === 1 ? 1 - confidence : confidence,
      color: '#4ecdc4',
      description: prediction === 1 ? 'Lower probability of not needing treatment' : 'Model suggests treatment may not be immediately necessary'
    },
  ];

  const featureData = Object.entries(feature_importance || {}).map(([feature, importance]) => ({
    feature: feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    rawFeature: feature,
    importance: importance,
    percentage: (importance * 100).toFixed(1),
    impact: importance > 0.2 ? 'High' : importance > 0.1 ? 'Medium' : 'Low'
  })).sort((a, b) => b.importance - a.importance);

  // Risk assessment data
  const riskFactors = featureData.slice(0, 6).map(item => ({
    factor: item.feature,
    score: Math.min(item.importance * 5, 5), // Scale to 0-5
    maxScore: 5
  }));


  const getResultIcon = () => prediction === 1 ? <Warning /> : <CheckCircle />;
  
  const getConfidenceLevel = () => {
    if (confidence >= 0.8) return 'Very High';
    if (confidence >= 0.7) return 'High';
    if (confidence >= 0.6) return 'Moderate';
    if (confidence >= 0.5) return 'Low';
    return 'Very Low';
  };

  const getConfidenceColor = () => {
    if (confidence >= 0.7) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getRiskLevel = () => {
    if (prediction === 1 && confidence >= 0.8) return 'High Priority';
    if (prediction === 1 && confidence >= 0.6) return 'Moderate Priority';
    if (prediction === 1) return 'Low Priority';
    return 'Minimal Risk';
  };

  const getRiskLevelColor = () => {
    const risk = getRiskLevel();
    if (risk === 'High Priority') return 'error';
    if (risk === 'Moderate Priority') return 'warning';
    if (risk === 'Low Priority') return 'info';
    return 'success';
  };

  const getRecommendations = () => {
    const recommendations = [];
    
    if (prediction === 1) {
      if (confidence >= 0.8) {
        recommendations.push({
          priority: 'High',
          action: 'Consult a Mental Health Professional',
          description: 'Schedule an appointment with a licensed therapist or counselor within the next week.',
          icon: <LocalHospital />,
          color: 'error'
        });
      }
      
      recommendations.push(
        {
          priority: 'Medium',
          action: 'Contact Support Resources',
          description: 'Reach out to mental health helplines or support groups in your area.',
          icon: <Phone />,
          color: 'warning'
        },
        {
          priority: 'Medium',
          action: 'Develop Coping Strategies',
          description: 'Practice stress management techniques like meditation, exercise, or journaling.',
          icon: <SelfImprovement />,
          color: 'info'
        }
      );
    } else {
      recommendations.push(
        {
          priority: 'Low',
          action: 'Maintain Mental Wellness',
          description: 'Continue healthy habits and stay aware of your mental health status.',
          icon: <HealthAndSafety />,
          color: 'success'
        },
        {
          priority: 'Low',
          action: 'Stay Informed',
          description: 'Learn about mental health awareness and prevention strategies.',
          icon: <School />,
          color: 'info'
        }
      );
    }
    
    return recommendations;
  };

  const getEducationalContent = () => {
    return {
      understanding: [
        "Mental health predictions are based on patterns found in large datasets of survey responses.",
        "The model considers multiple factors including personal history, current symptoms, and environmental factors.",
        "A higher confidence score means the model found stronger patterns matching your responses.",
        "These predictions are screening tools, not diagnostic instruments."
      ],
      factors: [
        "Family History: Genetic and environmental factors can influence mental health predisposition.",
        "Work Environment: Job stress, employment status, and workplace culture affect mental wellbeing.",
        "Social Support: Having strong social connections is protective for mental health.",
        "Coping Mechanisms: How you handle stress significantly impacts your mental health risk.",
        "Current Symptoms: Present mood, behavioral changes, and stress levels are key indicators."
      ],
      nextSteps: prediction === 1 ? [
        "Don't panic - seeking help is a sign of strength, not weakness.",
        "Mental health treatment is highly effective for most conditions.",
        "Early intervention often leads to better outcomes.",
        "Professional help can provide personalized strategies for your situation."
      ] : [
        "Continue maintaining good mental health practices.",
        "Stay aware of changes in your mood or behavior.",
        "Build and maintain strong social connections.",
        "Keep stress management techniques in your routine."
      ]
    };
  };

  const customTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Paper sx={{ p: 2, maxWidth: 200 }}>
          <Typography variant="body2" fontWeight="bold">{label}</Typography>
          <Typography variant="body2" color="primary">
            Importance: {(payload[0].value * 100).toFixed(1)}%
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  return (
    <Box sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom color="primary" fontWeight="bold">
          Mental Health Assessment Results
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
          Comprehensive Analysis & Recommendations
        </Typography>
        <Button 
          variant="outlined" 
          onClick={onReset}
          sx={{ mr: 2 }}
        >
          Take New Assessment
        </Button>
        <Button 
          variant="outlined" 
          onClick={() => setShowDetailedAnalysis(!showDetailedAnalysis)}
          startIcon={<Analytics />}
        >
          {showDetailedAnalysis ? 'Hide' : 'Show'} Detailed Analysis
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* Main Result - Enhanced */}
        <Grid item xs={12} lg={8}>
          <Card elevation={4} sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
            <CardContent sx={{ p: 4 }}>
              <Box display="flex" alignItems="center" mb={3}>
                <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', mr: 2, width: 56, height: 56 }}>
                  {getResultIcon()}
                </Avatar>
                <Box>
                  <Typography variant="h4" component="h2" fontWeight="bold">
                    {prediction_label}
                  </Typography>
                  <Typography variant="h6" sx={{ opacity: 0.9 }}>
                    Risk Level: {getRiskLevel()}
                  </Typography>
                </Box>
              </Box>
              
              <Typography variant="h6" sx={{ mb: 2, lineHeight: 1.6 }}>
                {prediction === 1 
                  ? "Our analysis suggests that mental health support could be beneficial for you. This recommendation is based on patterns identified in your responses."
                  : "Based on your responses, you appear to have good mental health indicators. Continue maintaining your current positive practices."
                }
              </Typography>

              <Box sx={{ display: 'flex', alignItems: 'center', mt: 3 }}>
                <AccessTime sx={{ mr: 1 }} />
                <Typography variant="body2">
                  Assessment completed: {new Date(timestamp).toLocaleString()}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Confidence & Risk Meter */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            <Card elevation={3}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <TrendingUp color="primary" />
                  <Typography variant="h6" sx={{ ml: 1 }}>
                    Confidence Score
                  </Typography>
                </Box>
                <Box textAlign="center">
                  <Typography variant="h2" color={getConfidenceColor()} fontWeight="bold">
                    {(confidence * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body1" color="text.secondary" gutterBottom>
                    {getConfidenceLevel()} Confidence
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={confidence * 100}
                    color={getConfidenceColor()}
                    sx={{ height: 12, borderRadius: 6, mb: 2 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    Model certainty in this prediction
                  </Typography>
                </Box>
              </CardContent>
            </Card>

            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Priority Level
                </Typography>
                <Chip
                  label={getRiskLevel()}
                  color={getRiskLevelColor()}
                  size="large"
                  sx={{ fontSize: '1rem', fontWeight: 'bold', width: '100%' }}
                />
              </CardContent>
            </Card>
          </Stack>
        </Grid>

        {/* Enhanced Visualizations */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Prediction Breakdown
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${(value * 100).toFixed(1)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <RechartsTooltip 
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <Paper sx={{ p: 2, maxWidth: 200 }}>
                            <Typography variant="body2" fontWeight="bold">{data.name}</Typography>
                            <Typography variant="body2">{(data.value * 100).toFixed(1)}%</Typography>
                            <Typography variant="caption" color="text.secondary">
                              {data.description}
                            </Typography>
                          </Paper>
                        );
                      }
                      return null;
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Risk Factor Radar Chart */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Factor Analysis
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={riskFactors}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="factor" tick={{ fontSize: 10 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 5]} />
                  <Radar
                    name="Risk Level"
                    dataKey="score"
                    stroke="#8884d8"
                    fill="#8884d8"
                    fillOpacity={0.3}
                  />
                  <RechartsTooltip content={customTooltip} />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Detailed Feature Importance */}
        <Grid item xs={12}>
          <Card elevation={3}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Assessment color="primary" />
                <Typography variant="h6" sx={{ ml: 1 }}>
                  Factor Impact Analysis
                </Typography>
                <Tooltip title="These factors had the most influence on your prediction">
                  <IconButton size="small" sx={{ ml: 1 }}>
                    <Info fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>

              <Alert severity="info" sx={{ mb: 3 }}>
                <Typography variant="body2">
                  <strong>How to read this chart:</strong> Higher bars indicate factors that had more influence on your prediction. 
                  The importance scores are relative - even small values can be significant in the overall prediction.
                </Typography>
              </Alert>

              {featureData.length > 0 ? (
                <>
                  <ResponsiveContainer width="100%" height={500}>
                    <BarChart 
                      data={featureData.slice(0, 8)} 
                      margin={{ top: 20, right: 50, left: 150, bottom: 60 }}
                      layout="horizontal"
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        type="number" 
                        domain={[0, 'dataMax']}
                        tickFormatter={(value) => (value * 100).toFixed(1) + '%'}
                        label={{ value: 'Relative Importance (%)', position: 'insideBottom', offset: -10 }}
                      />
                      <YAxis 
                        dataKey="feature" 
                        type="category" 
                        width={140}
                        tick={{ fontSize: 13 }}
                      />
                      <RechartsTooltip 
                        content={({ active, payload, label }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0];
                            return (
                              <Paper sx={{ p: 2, maxWidth: 300, boxShadow: 3 }}>
                                <Typography variant="body1" fontWeight="bold" gutterBottom>
                                  {label}
                                </Typography>
                                <Typography variant="body2" color="primary">
                                  Importance: {(data.value * 100).toFixed(2)}%
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Impact Level: {data.payload.impact}
                                </Typography>
                                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                                  This factor contributed {(data.value * 100).toFixed(1)}% to the overall prediction confidence.
                                </Typography>
                              </Paper>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar 
                        dataKey="importance" 
                        fill="#8884d8" 
                        radius={[0, 4, 4, 0]}
                      >
                        {featureData.slice(0, 8).map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={
                              entry.importance > 0.2 ? '#f44336' : 
                              entry.importance > 0.1 ? '#ff9800' : 
                              entry.importance > 0.05 ? '#2196f3' : '#4caf50'
                            } 
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>

                  <Box mt={4}>
                    <Typography variant="h6" gutterBottom fontWeight="bold">
                      Factor Impact Summary
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Understanding what influenced your prediction:
                    </Typography>
                    
                    <Grid container spacing={2}>
                      {featureData.slice(0, 8).map((item, index) => (
                        <Grid item xs={12} sm={6} md={4} lg={3} key={item.feature}>
                          <Paper 
                            sx={{ 
                              p: 2, 
                              bgcolor: 'grey.50',
                              border: '2px solid',
                              borderColor: 
                                item.importance > 0.2 ? 'error.light' : 
                                item.importance > 0.1 ? 'warning.light' : 
                                item.importance > 0.05 ? 'info.light' : 'success.light',
                              '&:hover': {
                                boxShadow: 3,
                                transform: 'translateY(-2px)',
                                transition: 'all 0.2s ease-in-out'
                              }
                            }}
                          >
                            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                              <Typography variant="body2" fontWeight="bold" noWrap title={item.feature}>
                                {item.feature.length > 15 ? `${item.feature.substring(0, 15)}...` : item.feature}
                              </Typography>
                              <Chip 
                                label={item.impact} 
                                size="small" 
                                color={
                                  item.importance > 0.2 ? 'error' : 
                                  item.importance > 0.1 ? 'warning' : 
                                  item.importance > 0.05 ? 'info' : 'success'
                                }
                              />
                            </Box>
                            
                            <Box display="flex" alignItems="center" mb={1}>
                              <LinearProgress
                                variant="determinate"
                                value={(item.importance / Math.max(...featureData.map(f => f.importance))) * 100}
                                sx={{ 
                                  flexGrow: 1, 
                                  mr: 1, 
                                  height: 8, 
                                  borderRadius: 4,
                                  bgcolor: 'grey.200'
                                }}
                                color={
                                  item.importance > 0.2 ? 'error' : 
                                  item.importance > 0.1 ? 'warning' : 
                                  item.importance > 0.05 ? 'info' : 'success'
                                }
                              />
                              <Typography variant="caption" fontWeight="bold">
                                {item.percentage}%
                              </Typography>
                            </Box>
                            
                            <Typography variant="caption" color="text.secondary">
                              Relative influence on prediction
                            </Typography>
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>

                  <Box mt={4}>
                    <Typography variant="h6" gutterBottom fontWeight="bold">
                      What These Numbers Mean
                    </Typography>
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, bgcolor: 'info.50' }}>
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="info.dark">
                            Impact Levels Explained:
                          </Typography>
                          <List dense>
                            <ListItem sx={{ py: 0.5 }}>
                              <Chip label="High" color="error" size="small" sx={{ mr: 1, minWidth: 60 }} />
                              <ListItemText 
                                primary="Major influence (>20%)" 
                                secondary="Primary driver of your prediction"
                                sx={{ '& .MuiListItemText-primary': { fontSize: '0.875rem' } }}
                              />
                            </ListItem>
                            <ListItem sx={{ py: 0.5 }}>
                              <Chip label="Medium" color="warning" size="small" sx={{ mr: 1, minWidth: 60 }} />
                              <ListItemText 
                                primary="Moderate influence (10-20%)" 
                                secondary="Important contributing factor"
                                sx={{ '& .MuiListItemText-primary': { fontSize: '0.875rem' } }}
                              />
                            </ListItem>
                            <ListItem sx={{ py: 0.5 }}>
                              <Chip label="Low" color="success" size="small" sx={{ mr: 1, minWidth: 60 }} />
                              <ListItemText 
                                primary="Minor influence (<10%)" 
                                secondary="Supporting information"
                                sx={{ '& .MuiListItemText-primary': { fontSize: '0.875rem' } }}
                              />
                            </ListItem>
                          </List>
                        </Paper>
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, bgcolor: 'success.50' }}>
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="success.dark">
                            Key Insights:
                          </Typography>
                          <List dense>
                            <ListItem sx={{ py: 0.5 }}>
                              <ListItemText 
                                primary="Most Important Factor" 
                                secondary={featureData[0]?.feature || 'N/A'}
                                sx={{ '& .MuiListItemText-primary': { fontSize: '0.875rem', fontWeight: 'bold' } }}
                              />
                            </ListItem>
                            <ListItem sx={{ py: 0.5 }}>
                              <ListItemText 
                                primary="Top 3 Factors Combined" 
                                secondary={`${(featureData.slice(0, 3).reduce((sum, item) => sum + item.importance, 0) * 100).toFixed(1)}% of total influence`}
                                sx={{ '& .MuiListItemText-primary': { fontSize: '0.875rem', fontWeight: 'bold' } }}
                              />
                            </ListItem>
                            <ListItem sx={{ py: 0.5 }}>
                              <ListItemText 
                                primary="Prediction Complexity" 
                                secondary={featureData.length > 5 ? 'Multiple factors considered' : 'Few key factors'}
                                sx={{ '& .MuiListItemText-primary': { fontSize: '0.875rem', fontWeight: 'bold' } }}
                              />
                            </ListItem>
                          </List>
                        </Paper>
                      </Grid>
                    </Grid>
                  </Box>
                </>
              ) : (
                <Alert severity="warning">
                  <Typography variant="body2">
                    Feature importance data is not available for this prediction. This may occur if the model 
                    doesn't support feature importance calculation or if there was an error in processing.
                  </Typography>
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recommendations */}
        <Grid item xs={12}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <LightbulbOutlined sx={{ mr: 1, verticalAlign: 'middle' }} />
                Personalized Recommendations
              </Typography>
              
              <Grid container spacing={3}>
                {getRecommendations().map((rec, index) => (
                  <Grid item xs={12} md={6} key={index}>
                    <Paper 
                      sx={{ 
                        p: 3, 
                        border: `2px solid`,
                        borderColor: `${rec.color}.light`,
                        bgcolor: `${rec.color}.50`,
                        '&:hover': { bgcolor: `${rec.color}.100` }
                      }}
                    >
                      <Box display="flex" alignItems="flex-start" mb={2}>
                        <Avatar sx={{ bgcolor: `${rec.color}.main`, mr: 2 }}>
                          {rec.icon}
                        </Avatar>
                        <Box>
                          <Typography variant="h6" fontWeight="bold">
                            {rec.action}
                          </Typography>
                          <Chip 
                            label={`${rec.priority} Priority`} 
                            size="small" 
                            color={rec.color}
                            sx={{ mt: 0.5 }}
                          />
                        </Box>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {rec.description}
                      </Typography>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Educational Content */}
        <Grid item xs={12}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <MenuBook sx={{ mr: 1, verticalAlign: 'middle' }} />
                Understanding Your Results
              </Typography>
              
              {Object.entries(getEducationalContent()).map(([section, content]) => (
                <Accordion key={section} sx={{ mb: 1 }}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="subtitle1" fontWeight="bold">
                      {section === 'understanding' ? 'How This Works' : 
                       section === 'factors' ? 'Key Factors Explained' : 'Next Steps'}
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <List>
                      {content.map((item, index) => (
                        <ListItem key={index} sx={{ pl: 0 }}>
                          <ListItemText 
                            primary={item}
                            sx={{ '& .MuiListItemText-primary': { fontSize: '0.95rem' } }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Crisis Resources */}
        {prediction === 1 && confidence >= 0.7 && (
          <Grid item xs={12}>
            <Alert severity="warning" sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                <Support sx={{ mr: 1, verticalAlign: 'middle' }} />
                Crisis Resources
              </Typography>
              <Typography variant="body1" paragraph>
                If you're experiencing thoughts of self-harm or suicide, please reach out immediately:
              </Typography>
              <Box>
                <Typography variant="body2">
                  • <strong>National Suicide Prevention Lifeline:</strong> 988 (US)
                </Typography>
                <Typography variant="body2">
                  • <strong>Crisis Text Line:</strong> Text HOME to 741741
                </Typography>
                <Typography variant="body2">
                  • <strong>Emergency Services:</strong> 911 (US) or your local emergency number
                </Typography>
              </Box>
            </Alert>
          </Grid>
        )}

        {/* Detailed Analysis Section */}
        <Collapse in={showDetailedAnalysis} sx={{ width: '100%' }}>
          <Grid item xs={12}>
            <Card elevation={3} sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <Analytics sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Technical Analysis Details
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      Model Performance Metrics
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText 
                          primary="Model Type" 
                          secondary="Random Forest Classifier"
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Training Accuracy" 
                          secondary="~74% (Cross-validated)"
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Features Analyzed" 
                          secondary={`${featureData.length} factors`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Dataset Size" 
                          secondary="290K+ responses"
                        />
                      </ListItem>
                    </List>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      Prediction Reliability
                    </Typography>
                    <Box>
                      <Typography variant="body2" gutterBottom>
                        Confidence Distribution:
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={confidence * 100} 
                        sx={{ height: 8, borderRadius: 4, mb: 1 }}
                      />
                      <Typography variant="caption" color="text.secondary">
                        Based on similarity to training patterns
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Collapse>

        {/* Disclaimer - Enhanced */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 3, bgcolor: 'warning.50', border: '1px solid', borderColor: 'warning.light' }}>
            <Alert severity="warning" sx={{ bgcolor: 'transparent' }}>
              <Typography variant="h6" gutterBottom>
                Medical Disclaimer
              </Typography>
              <Typography variant="body2" paragraph>
                This assessment is a screening tool based on machine learning analysis and is <strong>not a medical diagnosis</strong>. 
                Results should not replace professional medical advice, diagnosis, or treatment.
              </Typography>
              <Typography variant="body2">
                If you are experiencing mental health concerns, please consult with a qualified healthcare professional, 
                therapist, or counselor. Mental health professionals can provide personalized assessment and evidence-based treatment options.
              </Typography>
            </Alert>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ResultsDisplay; 