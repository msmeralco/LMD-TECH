"""
GhostLoad Mapper Backend - Startup Script
==========================================

Run this to start the FastAPI backend server.

Usage:
    python start_server.py

The server will start at http://localhost:8000
API docs available at http://localhost:8000/docs
"""

import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Now import and run
from app.main import app
import uvicorn

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 GHOSTLOAD MAPPER BACKEND SERVER")
    print("="*70)
    print(f"\n📂 Backend directory: {backend_dir}")
    print(f"🌐 Server: http://localhost:8000")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print(f"🏥 Health Check: http://localhost:8000/api/health")
    print("\n" + "="*70)
    print("Starting server...\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
