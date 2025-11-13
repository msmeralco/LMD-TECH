import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """App configuration from environment variables"""
    PROJECT_NAME: str = "GhostLoadMapper API"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Transformer-aware Low-Load Anomaly Detection"
    
    # Firebase/Firestore config
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
    FIREBASE_CREDENTIALS_PATH: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
    
    # API config
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",      # React dev server
        "http://localhost:5173",      # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

settings = Settings()