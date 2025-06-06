# Deployment Alternatives Guide

Since Railway deployment isn't working, here are several reliable alternatives:

## Option 1: Vercel (Frontend) + Render (Backend) [RECOMMENDED]

### Backend on Render:
1. Go to [render.com](https://render.com) and create an account
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3
   - **Root Directory**: `backend`
5. Add environment variables:
   - `SECRET_KEY`: Generate a secure key
   - `ALGORITHM`: `HS256`
   - `ACCESS_TOKEN_EXPIRE_MINUTES`: `30`
6. Deploy and note your backend URL (e.g., `https://your-app.onrender.com`)

### Frontend on Vercel:
1. Go to [vercel.com](https://vercel.com) and create an account
2. Click "New Project" and import your repository
3. Configure:
   - **Framework Preset**: Create React App
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`
4. Add environment variable:
   - `REACT_APP_API_URL`: Your Render backend URL
5. Deploy

## Option 2: Heroku (Full Stack)

### Backend:
```bash
# Install Heroku CLI and login
heroku login

# Create app for backend
cd backend
heroku create your-mental-health-backend

# Set environment variables
heroku config:set SECRET_KEY=your-secret-key
heroku config:set ALGORITHM=HS256
heroku config:set ACCESS_TOKEN_EXPIRE_MINUTES=30

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### Frontend:
```bash
# Create app for frontend
cd frontend
heroku create your-mental-health-frontend

# Add buildpack
heroku buildpacks:set https://github.com/mars/create-react-app-buildpack.git

# Set API URL
heroku config:set REACT_APP_API_URL=https://your-mental-health-backend.herokuapp.com

# Deploy
git add .
git commit -m "Deploy frontend to Heroku"
git push heroku main
```

## Option 3: Netlify (Frontend) + Heroku (Backend)

### Backend on Heroku:
Follow the backend steps from Option 2

### Frontend on Netlify:
1. Go to [netlify.com](https://netlify.com) and create an account
2. Click "New site from Git"
3. Choose your repository
4. Configure:
   - **Base directory**: `frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `frontend/build`
5. Add environment variable:
   - `REACT_APP_API_URL`: Your Heroku backend URL
6. Deploy

## Option 4: Local Docker Deployment

### Prerequisites:
- Docker and Docker Compose installed

### Deployment:
```bash
# Build and run all services
docker-compose up --build

# Access:
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### For production Docker deployment:
```bash
# Build for production
docker-compose -f docker-compose.prod.yml up --build -d
```

## Option 5: DigitalOcean App Platform

1. Go to [digitalocean.com](https://digitalocean.com/products/app-platform)
2. Create account and new app
3. Connect GitHub repository
4. Configure components:
   - **Backend**: Python app, source: `backend/`, run command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Frontend**: Static site, source: `frontend/`, build command: `npm run build`, output: `build/`
5. Set environment variables
6. Deploy

## Recommended Choice

**For beginners**: Option 1 (Vercel + Render) - Free, reliable, easy setup
**For developers**: Option 4 (Docker) - Full control, works everywhere
**For production**: Option 5 (DigitalOcean) - Scalable, professional

## Next Steps

1. Choose your preferred option
2. Follow the specific deployment steps
3. Update the API URL in your frontend configuration
4. Test the deployed application

## Common Issues and Solutions

### Backend Issues:
- **Memory errors**: Use lighter ML models or increase memory limits
- **Timeout errors**: Optimize model loading and prediction speed
- **CORS errors**: Ensure CORS is properly configured in FastAPI

### Frontend Issues:
- **API connection**: Verify the API URL is correct and accessible
- **Build errors**: Check node version compatibility
- **Static files**: Ensure proper routing configuration

## Support

If you encounter issues with any deployment method, the logs will help identify the problem:
- Render: Check service logs in dashboard
- Vercel: Check function logs and build logs
- Heroku: Use `heroku logs --tail`
- Docker: Use `docker-compose logs` 