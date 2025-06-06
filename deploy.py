#!/usr/bin/env python3
"""
Mental Health Prediction System - Deployment Helper
This script helps prepare the project for deployment to Railway.app
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def print_banner():
    """Print deployment banner"""
    print("=" * 60)
    print("🚀 Mental Health Prediction System - Deployment Helper")
    print("=" * 60)
    print()

def check_requirements():
    """Check if all requirements are met"""
    print("📋 Checking deployment requirements...")
    
    # Check if git is installed
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        print("✅ Git is installed")
    except:
        print("❌ Git is not installed or not in PATH")
        return False
    
    # Check if we're in a git repository
    if not os.path.exists(".git"):
        print("❌ Not in a git repository")
        return False
    else:
        print("✅ Git repository detected")
    
    # Check for essential files
    essential_files = [
        "backend/app/main.py",
        "frontend/package.json",
        "backend/requirements.txt",
        "data/Mental Health Dataset.csv"
    ]
    
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} found")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def prepare_deployment():
    """Prepare files for deployment"""
    print("\n🔧 Preparing deployment files...")
    
    # Create .env.example for backend
    backend_env = """# Production Environment Variables
ENVIRONMENT=production
PORT=8000
CORS_ORIGINS=*
"""
    
    with open("backend/.env.example", "w") as f:
        f.write(backend_env)
    print("✅ Created backend/.env.example")
    
    # Create .env.example for frontend
    frontend_env = """# Frontend Environment Variables
REACT_APP_API_URL=https://your-backend-url.railway.app
"""
    
    with open("frontend/.env.example", "w") as f:
        f.write(frontend_env)
    print("✅ Created frontend/.env.example")
    
    # Update package.json scripts for deployment
    package_json_path = "frontend/package.json"
    if os.path.exists(package_json_path):
        with open(package_json_path, "r") as f:
            package_data = json.load(f)
        
        package_data["scripts"]["build"] = "react-scripts build"
        package_data["scripts"]["start"] = "serve -s build -l 3000"
        
        # Add serve dependency
        if "dependencies" not in package_data:
            package_data["dependencies"] = {}
        package_data["dependencies"]["serve"] = "^14.2.0"
        
        with open(package_json_path, "w") as f:
            json.dump(package_data, f, indent=2)
        print("✅ Updated frontend/package.json for production")

def show_deployment_options():
    """Show deployment options"""
    print("\n🚀 Deployment Options:")
    print()
    
    print("1️⃣  RAILWAY.APP (Recommended - FREE)")
    print("   • $5/month free credit")
    print("   • Easy GitHub integration")
    print("   • Automatic HTTPS")
    print("   • Deploy both frontend & backend")
    print("   • 24/7 uptime")
    print()
    
    print("2️⃣  VERCEL + RAILWAY")
    print("   • Vercel: Frontend (free)")
    print("   • Railway: Backend (free tier)")
    print("   • Excellent performance")
    print()
    
    print("3️⃣  RENDER.COM")
    print("   • Free tier available")
    print("   • Full-stack deployment")
    print("   • Auto-deploy from GitHub")
    print()

def show_railway_instructions():
    """Show detailed Railway deployment instructions"""
    print("\n📝 STEP-BY-STEP RAILWAY DEPLOYMENT:")
    print()
    
    print("🔗 1. Setup Railway Account:")
    print("   → Visit: https://railway.app")
    print("   → Sign up with GitHub (free)")
    print("   → Connect your repository")
    print()
    
    print("🖥️  2. Deploy Backend:")
    print("   → Click 'New Project'")
    print("   → Select 'Deploy from GitHub repo'")
    print("   → Choose this repository")
    print("   → Set Root Directory: 'backend'")
    print("   → Add environment variables:")
    print("     ENVIRONMENT=production")
    print("     PORT=8000")
    print("   → Deploy!")
    print()
    
    print("🌐 3. Deploy Frontend:")
    print("   → In same project, click 'Add Service'")
    print("   → Select 'GitHub repo'")
    print("   → Choose same repository")
    print("   → Set Root Directory: 'frontend'")
    print("   → Add environment variable:")
    print("     REACT_APP_API_URL=https://your-backend-url.railway.app")
    print("   → Deploy!")
    print()
    
    print("✅ 4. Verify Deployment:")
    print("   → Test backend: https://your-backend-url.railway.app/health")
    print("   → Test frontend: https://your-frontend-url.railway.app")
    print("   → Check API docs: https://your-backend-url.railway.app/docs")
    print()

def commit_changes():
    """Commit deployment preparation changes"""
    print("\n📤 Committing deployment preparation...")
    
    try:
        # Add all changes
        subprocess.run(["git", "add", "."], check=True)
        
        # Commit changes
        commit_message = "Deploy: Prepare project for Railway.app deployment"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        print("✅ Changes committed successfully")
        print("📤 Push to GitHub: git push origin main")
        
    except subprocess.CalledProcessError:
        print("ℹ️  No new changes to commit")

def main():
    """Main deployment helper function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please fix the requirements above before proceeding.")
        sys.exit(1)
    
    # Prepare deployment
    prepare_deployment()
    
    # Show deployment options
    show_deployment_options()
    
    # Show Railway instructions
    show_railway_instructions()
    
    # Commit changes
    commit_changes()
    
    print("\n🎉 PROJECT READY FOR DEPLOYMENT!")
    print()
    print("🔗 Quick Links:")
    print("   • Railway: https://railway.app")
    print("   • Vercel: https://vercel.com")
    print("   • Render: https://render.com")
    print()
    print("📖 Detailed Guide: See DEPLOYMENT.md")
    print()
    print("💡 TIP: Railway.app is the easiest option for this project!")
    print()

if __name__ == "__main__":
    main() 