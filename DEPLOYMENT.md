# Deployment Guide: Mental Health Prediction System on Vercel

## Prerequisites

1. **Git Repository**: Push your code to GitHub
2. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
3. **Node.js**: Version 18+ recommended

## Step-by-Step Deployment

### 1. Prepare Your Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial deployment setup"

# Add remote repository (replace with your GitHub repo)
git remote add origin https://github.com/yourusername/mental-health-prediction.git

# Push to GitHub
git push -u origin main
```

### 2. Deploy to Vercel

#### Option A: Using Vercel Dashboard (Recommended)
1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click "New Project"
3. Import your GitHub repository
4. Vercel will automatically detect the configuration from `vercel.json`
5. Click "Deploy"

#### Option B: Using Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy from your project directory
vercel --prod
```

### 3. Environment Variables (Optional)

In your Vercel dashboard, you can set environment variables:
- `SECRET_KEY`: Random string for JWT tokens
- `ALGORITHM`: HS256 (default)
- `ACCESS_TOKEN_EXPIRE_MINUTES`: 30 (default)

### 4. Custom Domain (Optional)

1. In Vercel dashboard, go to your project
2. Click "Settings" → "Domains"
3. Add your custom domain

## Project Structure

```
mental-health-prediction/
├── frontend/                 # React application
│   ├── src/
│   ├── public/
│   └── package.json
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── main.py          # Main FastAPI app
│   │   └── fallback_model.py # Fallback model
│   └── requirements-vercel.txt
├── models/                   # Trained ML models
│   ├── best_model.pkl       # Main model (346MB)
│   └── preprocessor.pkl     # Data preprocessor
├── vercel.json              # Vercel configuration
└── package.json             # Root package.json
```

## API Endpoints

Once deployed, your API will be available at:
- `https://your-app.vercel.app/api/health` - Health check
- `https://your-app.vercel.app/api/predict` - Prediction endpoint
- `https://your-app.vercel.app/api/model-info` - Model information

## Frontend Routes

- `/` - Home page with prediction form
- `/about` - About page with project information

## Troubleshooting

### Common Issues

1. **Build Timeout**: Large model files (346MB) might cause issues
   - Solution: Consider using model compression or external storage

2. **Memory Limits**: Vercel has memory limits for serverless functions
   - Solution: Fallback model is implemented for this case

3. **CORS Issues**: If frontend can't connect to API
   - Check `allow_origins` in `main.py`
   - Ensure API routes are configured correctly

### Debugging

Check Vercel function logs:
1. Go to Vercel dashboard
2. Click on your project
3. Go to "Functions" tab
4. Click on any function to see logs

## Performance Optimization

1. **Model Size**: The trained model is 346MB, which is large for serverless
2. **Fallback Model**: A rule-based fallback is implemented
3. **Cold Starts**: First request might be slower due to model loading

## Security Notes

- Default demo token: `demo-token-12345`
- In production, implement proper JWT authentication
- Consider rate limiting for the API

## Updates

To update your deployment:
1. Push changes to GitHub
2. Vercel will automatically redeploy
3. Or use `vercel --prod` command

## Cost

- Vercel Pro: Free tier includes:
  - 100GB bandwidth
  - Unlimited projects
  - Analytics
  - Custom domains

Your application should run comfortably within free tier limits. 