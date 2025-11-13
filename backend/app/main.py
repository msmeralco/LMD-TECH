from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import routes
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(routes.router)
logger.info("✅ API routes loaded")

@app.get("/")
def root():
    return {
        "message": "Welcome to GhostLoadMapper API",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "run_analysis": "POST /api/run",
            "get_results": "GET /api/results/{run_id}",
            "export_csv": "GET /api/export/{run_id}?level=transformer",
            "transformers_geojson": "GET /api/transformers.geojson?run_id={run_id}",
            "list_runs": "GET /api/runs",
        }
    }