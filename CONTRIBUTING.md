# Contributing to Mental Health Treatment Prediction System

We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

### Prerequisites

- Python 3.8+
- Node.js 14+
- Git

### Quick Setup

Run the setup script for your platform:

**Windows:**
```bash
setup.bat
```

**Unix/Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mental-health-prediction.git
   cd mental-health-prediction
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

## Code Style

### Python (Backend)

- Follow PEP 8 style guide
- Use type hints where possible
- Write docstrings for functions and classes
- Use meaningful variable and function names

### JavaScript/React (Frontend)

- Use ES6+ features
- Follow React best practices
- Use meaningful component and variable names
- Write comments for complex logic

### General Guidelines

- Keep functions small and focused
- Write self-documenting code
- Add comments for complex business logic
- Use consistent naming conventions

## Testing

### Backend Tests

```bash
cd backend
pytest tests/
```

### Frontend Tests

```bash
cd frontend
npm test
```

### Writing Tests

- Write unit tests for new functions
- Write integration tests for API endpoints
- Write component tests for React components
- Aim for good test coverage

## Project Structure

```
mental-health-prediction/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # Main FastAPI application
│   │   ├── data_preprocessing.py  # Data preprocessing
│   │   └── model_training.py      # ML model training
│   ├── tests/              # Backend tests
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   └── services/      # API services
│   └── package.json       # Node.js dependencies
├── data/                  # Dataset (not in git)
├── models/                # Trained models (not in git)
└── docs/                  # Documentation
```

## Reporting Bugs

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/mental-health-prediction/issues).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests

We use GitHub issues to track feature requests as well. Please provide:

- Clear description of the feature
- Use case and motivation
- Possible implementation approach
- Any relevant examples

## Guidelines for Specific Components

### Machine Learning Models

- Document model performance metrics
- Include cross-validation results
- Explain feature selection rationale
- Provide interpretability analysis

### API Development

- Follow RESTful principles
- Include proper error handling
- Add comprehensive documentation
- Write API tests

### Frontend Development

- Ensure responsive design
- Follow accessibility guidelines
- Optimize for performance
- Test across different browsers

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue with the question label or contact the maintainers directly.

---

Thank you for your interest in contributing to mental health technology! 🧠💙 