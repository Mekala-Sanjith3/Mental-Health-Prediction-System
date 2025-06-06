import React, { useState } from 'react';
import { useForm, Controller } from 'react-hook-form';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Box,
  Alert,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  Card,
  CardContent,
} from '@mui/material';
import { PsychologyOutlined, SendOutlined } from '@mui/icons-material';
import PredictionAPI from '../services/api';
import ResultsDisplay from './ResultsDisplay';

const steps = ['Personal Information', 'Mental Health History', 'Current Status'];

const PredictionForm = ({ onPredictionResult, predictionResult }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiKey] = useState('demo-token-12345');

  const {
    control,
    handleSubmit,
    formState: { errors },
    trigger,
  } = useForm({
    defaultValues: {
      Gender: '',
      Country: '',
      Occupation: '',
      self_employed: '',
      family_history: '',
      Days_Indoors: '',
      Growing_Stress: '',
      Changes_Habits: '',
      Mental_Health_History: '',
      Mood_Swings: '',
      Coping_Struggles: '',
      Work_Interest: '',
      Social_Weakness: '',
      mental_health_interview: '',
      care_options: '',
    }
  });

  const handleNext = async () => {
    const fields = getStepFields(activeStep);
    const isStepValid = await trigger(fields);
    if (isStepValid) {
      setActiveStep((prevActiveStep) => prevActiveStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const getStepFields = (step) => {
    switch (step) {
      case 0:
        return ['Gender', 'Country', 'Occupation', 'self_employed'];
      case 1:
        return ['family_history', 'Mental_Health_History', 'mental_health_interview'];
      case 2:
        return ['Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness', 'care_options'];
      default:
        return [];
    }
  };

  const onSubmit = async (data) => {
    setLoading(true);
    setError(null);

    try {
      const result = await PredictionAPI.predict(data, apiKey);
      onPredictionResult(result);
    } catch (err) {
      setError(err.message || 'An error occurred during prediction');
    } finally {
      setLoading(false);
    }
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Controller
                name="Gender"
                control={control}
                rules={{ required: 'Gender is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.Gender}>
                    <InputLabel id="gender-label">Gender</InputLabel>
                    <Select 
                      {...field} 
                      labelId="gender-label"
                      label="Gender"
                      value={field.value || ''}
                    >
                      <MenuItem value="Male">Male</MenuItem>
                      <MenuItem value="Female">Female</MenuItem>
                      <MenuItem value="Other">Other</MenuItem>
                    </Select>
                    {errors.Gender && (
                      <Typography variant="caption" color="error">
                        {errors.Gender.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="Country"
                control={control}
                rules={{ required: 'Country is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Country"
                    error={!!errors.Country}
                    helperText={errors.Country?.message}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="Occupation"
                control={control}
                rules={{ required: 'Occupation is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Occupation"
                    error={!!errors.Occupation}
                    helperText={errors.Occupation?.message}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="self_employed"
                control={control}
                rules={{ required: 'Self-employment status is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.self_employed}>
                    <InputLabel id="self-employed-label">Self Employed</InputLabel>
                    <Select 
                      {...field} 
                      labelId="self-employed-label"
                      label="Self Employed"
                      value={field.value || ''}
                    >
                      <MenuItem value="Yes">Yes</MenuItem>
                      <MenuItem value="No">No</MenuItem>
                      <MenuItem value="Unknown">Unknown</MenuItem>
                    </Select>
                    {errors.self_employed && (
                      <Typography variant="caption" color="error">
                        {errors.self_employed.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Controller
                name="family_history"
                control={control}
                rules={{ required: 'Family history is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.family_history}>
                    <InputLabel id="family-history-label">Family History of Mental Health</InputLabel>
                    <Select 
                      {...field} 
                      labelId="family-history-label"
                      label="Family History of Mental Health"
                      value={field.value || ''}
                    >
                      <MenuItem value="Yes">Yes</MenuItem>
                      <MenuItem value="No">No</MenuItem>
                    </Select>
                    {errors.family_history && (
                      <Typography variant="caption" color="error">
                        {errors.family_history.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="Mental_Health_History"
                control={control}
                rules={{ required: 'Mental health history is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.Mental_Health_History}>
                    <InputLabel id="mental-health-history-label">Personal Mental Health History</InputLabel>
                    <Select 
                      {...field} 
                      labelId="mental-health-history-label"
                      label="Personal Mental Health History"
                      value={field.value || ''}
                    >
                      <MenuItem value="Yes">Yes</MenuItem>
                      <MenuItem value="No">No</MenuItem>
                    </Select>
                    {errors.Mental_Health_History && (
                      <Typography variant="caption" color="error">
                        {errors.Mental_Health_History.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="mental_health_interview"
                control={control}
                rules={{ required: 'Mental health interview willingness is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.mental_health_interview}>
                    <InputLabel id="interview-label">Would you be willing to discuss mental health with a professional?</InputLabel>
                    <Select 
                      {...field} 
                      labelId="interview-label"
                      label="Would you be willing to discuss mental health with a professional?"
                      value={field.value || ''}
                    >
                      <MenuItem value="Yes">Yes</MenuItem>
                      <MenuItem value="No">No</MenuItem>
                    </Select>
                    {errors.mental_health_interview && (
                      <Typography variant="caption" color="error">
                        {errors.mental_health_interview.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
          </Grid>
        );

      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Controller
                name="Days_Indoors"
                control={control}
                rules={{ required: 'Days indoors is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.Days_Indoors}>
                    <InputLabel id="days-indoors-label">Days Spent Indoors</InputLabel>
                    <Select 
                      {...field} 
                      labelId="days-indoors-label"
                      label="Days Spent Indoors"
                      value={field.value || ''}
                    >
                      <MenuItem value="Go out Every day">Go out Every day</MenuItem>
                      <MenuItem value="1-14 days">1-14 days</MenuItem>
                      <MenuItem value="15-30 days">15-30 days</MenuItem>
                      <MenuItem value="31-60 days">31-60 days</MenuItem>
                      <MenuItem value="More than 2 months">More than 2 months</MenuItem>
                    </Select>
                    {errors.Days_Indoors && (
                      <Typography variant="caption" color="error">
                        {errors.Days_Indoors.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="Mood_Swings"
                control={control}
                rules={{ required: 'Mood swings frequency is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.Mood_Swings}>
                    <InputLabel id="mood-swings-label">Mood Swings Frequency</InputLabel>
                    <Select 
                      {...field} 
                      labelId="mood-swings-label"
                      label="Mood Swings Frequency"
                      value={field.value || ''}
                    >
                      <MenuItem value="Low">Low</MenuItem>
                      <MenuItem value="Medium">Medium</MenuItem>
                      <MenuItem value="High">High</MenuItem>
                    </Select>
                    {errors.Mood_Swings && (
                      <Typography variant="caption" color="error">
                        {errors.Mood_Swings.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="Growing_Stress"
                control={control}
                rules={{ required: 'Growing stress level is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.Growing_Stress}>
                    <InputLabel id="stress-label">Experiencing Growing Stress</InputLabel>
                    <Select 
                      {...field} 
                      labelId="stress-label"
                      label="Experiencing Growing Stress"
                      value={field.value || ''}
                    >
                      <MenuItem value="Yes">Yes</MenuItem>
                      <MenuItem value="No">No</MenuItem>
                    </Select>
                    {errors.Growing_Stress && (
                      <Typography variant="caption" color="error">
                        {errors.Growing_Stress.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="Changes_Habits"
                control={control}
                rules={{ required: 'Changes in habits is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.Changes_Habits}>
                    <InputLabel id="habits-label">Changes in Habits</InputLabel>
                    <Select 
                      {...field} 
                      labelId="habits-label"
                      label="Changes in Habits"
                      value={field.value || ''}
                    >
                      <MenuItem value="Yes">Yes</MenuItem>
                      <MenuItem value="No">No</MenuItem>
                    </Select>
                    {errors.Changes_Habits && (
                      <Typography variant="caption" color="error">
                        {errors.Changes_Habits.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="Coping_Struggles"
                control={control}
                rules={{ required: 'Coping struggles is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.Coping_Struggles}>
                    <InputLabel id="coping-label">Struggling to Cope</InputLabel>
                    <Select 
                      {...field} 
                      labelId="coping-label"
                      label="Struggling to Cope"
                      value={field.value || ''}
                    >
                      <MenuItem value="Yes">Yes</MenuItem>
                      <MenuItem value="No">No</MenuItem>
                    </Select>
                    {errors.Coping_Struggles && (
                      <Typography variant="caption" color="error">
                        {errors.Coping_Struggles.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="Work_Interest"
                control={control}
                rules={{ required: 'Work interest is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.Work_Interest}>
                    <InputLabel id="work-label">Interest in Work</InputLabel>
                    <Select 
                      {...field} 
                      labelId="work-label"
                      label="Interest in Work"
                      value={field.value || ''}
                    >
                      <MenuItem value="Yes">Yes</MenuItem>
                      <MenuItem value="No">No</MenuItem>
                    </Select>
                    {errors.Work_Interest && (
                      <Typography variant="caption" color="error">
                        {errors.Work_Interest.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="Social_Weakness"
                control={control}
                rules={{ required: 'Social weakness is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.Social_Weakness}>
                    <InputLabel id="social-label">Social Weakness</InputLabel>
                    <Select 
                      {...field} 
                      labelId="social-label"
                      label="Social Weakness"
                      value={field.value || ''}
                    >
                      <MenuItem value="Yes">Yes</MenuItem>
                      <MenuItem value="No">No</MenuItem>
                    </Select>
                    {errors.Social_Weakness && (
                      <Typography variant="caption" color="error">
                        {errors.Social_Weakness.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="care_options"
                control={control}
                rules={{ required: 'Care options awareness is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.care_options}>
                    <InputLabel id="care-options-label">Awareness of Care Options</InputLabel>
                    <Select 
                      {...field} 
                      labelId="care-options-label"
                      label="Awareness of Care Options"
                      value={field.value || ''}
                    >
                      <MenuItem value="Yes">Yes</MenuItem>
                      <MenuItem value="No">No</MenuItem>
                      <MenuItem value="Not sure">Not sure</MenuItem>
                    </Select>
                    {errors.care_options && (
                      <Typography variant="caption" color="error">
                        {errors.care_options.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />
            </Grid>
          </Grid>
        );

      default:
        return 'Unknown step';
    }
  };

  if (predictionResult) {
    return <ResultsDisplay result={predictionResult} onReset={() => onPredictionResult(null)} />;
  }

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <PsychologyOutlined sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
          <Typography variant="h4" component="h1" gutterBottom>
            Mental Health Assessment
          </Typography>
        </Box>

        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <form onSubmit={handleSubmit(onSubmit)}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                {steps[activeStep]}
              </Typography>
              {renderStepContent(activeStep)}
            </CardContent>
          </Card>

          <Box sx={{ display: 'flex', flexDirection: 'row', pt: 2 }}>
            <Button
              color="inherit"
              disabled={activeStep === 0}
              onClick={handleBack}
              sx={{ mr: 1 }}
            >
              Back
            </Button>
            <Box sx={{ flex: '1 1 auto' }} />
            {activeStep === steps.length - 1 ? (
              <Button
                type="submit"
                variant="contained"
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : <SendOutlined />}
              >
                {loading ? 'Analyzing...' : 'Get Prediction'}
              </Button>
            ) : (
              <Button onClick={handleNext} variant="contained">
                Next
              </Button>
            )}
          </Box>
        </form>
      </Paper>
    </Container>
  );
};

export default PredictionForm; 