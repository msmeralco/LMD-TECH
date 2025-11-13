"""
Firestore Database Helper Functions
====================================

Handles all Firebase/Firestore interactions for GhostLoad Mapper.
Stores ML pipeline results and serves data to frontend.

Author: Backend Team
Date: November 13, 2025
"""

import firebase_admin
from firebase_admin import credentials, firestore
from app.config import settings
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Initialize Firebase (only once)
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
        logger.info("✅ Firebase initialized successfully")
    except Exception as e:
        logger.error(f"❌ Firebase initialization failed: {e}")
        raise

# Get Firestore client
db = firestore.client()


# ==============================================================================
# RUN RESULTS MANAGEMENT
# ==============================================================================

async def save_run_results(run_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save complete ML pipeline results to Firestore.
    
    This stores:
    - City aggregations (for city-level zoom)
    - Barangay aggregations (for barangay-level zoom)
    - Transformer data (for transformer markers)
    - Meter predictions (for meter-level details)
    - High-risk summary (for sidebar alerts)
    
    Args:
        run_id: Unique identifier for this analysis run
        results: Dictionary containing all ML pipeline outputs
    
    Returns:
        Status dictionary
    """
    try:
        doc_ref = db.collection("runs").document(run_id)
        
        # Prepare document data
        doc_data = {
            "run_id": run_id,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "status": "completed",
            "total_meters": results.get("total_meters", 0),
            "total_transformers": results.get("total_transformers", 0),
            "total_barangays": results.get("total_barangays", 0),
            "total_cities": results.get("total_cities", 0),
            "high_risk_count": results.get("high_risk_count", 0),
            
            # Hierarchical data for frontend
            "cities": results.get("cities", []),
            "barangays": results.get("barangays", []),
            "transformers": results.get("transformers", []),
            "meters": results.get("meters", []),
            "high_risk_summary": results.get("high_risk_summary", {}),
            
            # Metadata
            "processing_time_seconds": results.get("processing_time", 0),
            "model_version": results.get("model_version", "1.0"),
        }
        
        # Save to Firestore
        doc_ref.set(doc_data)
        
        logger.info(f"✅ Saved results for run {run_id} ({results.get('total_meters', 0)} meters)")
        
        return {
            "run_id": run_id,
            "status": "saved",
            "total_meters": results.get("total_meters", 0),
            "total_transformers": results.get("total_transformers", 0),
        }
        
    except Exception as e:
        logger.error(f"❌ Error saving run results: {e}")
        raise


async def get_run_results(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve complete analysis results for a specific run.
    
    Args:
        run_id: The analysis run ID
    
    Returns:
        Dictionary with all results or None if not found
    """
    try:
        doc = db.collection("runs").document(run_id).get()
        
        if doc.exists:
            data = doc.to_dict()
            logger.info(f"✅ Retrieved results for run {run_id}")
            return data
        else:
            logger.warning(f"⚠️ Run {run_id} not found")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error retrieving run results: {e}")
        raise


async def list_recent_runs(limit: int = 10) -> List[Dict[str, Any]]:
    """
    List recent analysis runs.
    
    Args:
        limit: Number of runs to return
    
    Returns:
        List of run summaries
    """
    try:
        runs_ref = (
            db.collection("runs")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        
        docs = runs_ref.stream()
        
        runs = []
        for doc in docs:
            data = doc.to_dict()
            runs.append({
                "run_id": doc.id,
                "timestamp": data.get("timestamp"),
                "total_meters": data.get("total_meters", 0),
                "high_risk_count": data.get("high_risk_count", 0),
                "status": data.get("status", "unknown"),
            })
        
        logger.info(f"✅ Retrieved {len(runs)} recent runs")
        return runs
        
    except Exception as e:
        logger.error(f"❌ Error listing runs: {e}")
        raise


async def delete_run(run_id: str) -> Dict[str, str]:
    """
    Delete a run and its results.
    
    Args:
        run_id: The run ID to delete
    
    Returns:
        Status dictionary
    """
    try:
        db.collection("runs").document(run_id).delete()
        logger.info(f"✅ Deleted run {run_id}")
        return {"status": "deleted", "run_id": run_id}
        
    except Exception as e:
        logger.error(f"❌ Error deleting run: {e}")
        raise


# ==============================================================================
# METADATA MANAGEMENT
# ==============================================================================

async def save_transformer_metadata(transformers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Save transformer metadata (separate from run results).
    
    Useful for caching transformer locations, capacities, etc.
    
    Args:
        transformers: List of transformer dictionaries
    
    Returns:
        Status dictionary
    """
    try:
        doc_ref = db.collection("metadata").document("transformers")
        
        doc_ref.set({
            "data": transformers,
            "count": len(transformers),
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        
        logger.info(f"✅ Saved metadata for {len(transformers)} transformers")
        
        return {
            "status": "saved",
            "transformer_count": len(transformers),
        }
        
    except Exception as e:
        logger.error(f"❌ Error saving transformer metadata: {e}")
        raise


async def get_transformer_metadata() -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve transformer metadata.
    
    Returns:
        List of transformers or None
    """
    try:
        doc = db.collection("metadata").document("transformers").get()
        
        if doc.exists:
            data = doc.to_dict()
            return data.get("data", [])
        else:
            return None
            
    except Exception as e:
        logger.error(f"❌ Error retrieving transformer metadata: {e}")
        raise


# ==============================================================================
# UPLOAD TRACKING (Optional)
# ==============================================================================

async def save_upload_metadata(upload_id: str, metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Save metadata about uploaded CSV files.
    
    Args:
        upload_id: Unique upload identifier
        metadata: Upload metadata (filename, row count, columns, etc.)
    
    Returns:
        Status dictionary
    """
    try:
        doc_ref = db.collection("uploads").document(upload_id)
        
        doc_ref.set({
            **metadata,
            "upload_time": firestore.SERVER_TIMESTAMP,
        })
        
        logger.info(f"✅ Saved upload metadata for {upload_id}")
        
        return {"upload_id": upload_id, "status": "saved"}
        
    except Exception as e:
        logger.error(f"❌ Error saving upload metadata: {e}")
        raise


# ==============================================================================
# HEALTH CHECK
# ==============================================================================

async def health_check() -> Dict[str, str]:
    """
    Check if Firestore connection is healthy.
    
    Returns:
        Status dictionary
    """
    try:
        # Try a simple write and read
        test_ref = db.collection("_health").document("test")
        test_ref.set({"timestamp": firestore.SERVER_TIMESTAMP})
        
        doc = test_ref.get()
        
        if doc.exists:
            return {"status": "healthy", "service": "Firestore"}
        else:
            return {"status": "unhealthy", "service": "Firestore"}
            
    except Exception as e:
        logger.error(f"❌ Firestore health check failed: {e}")
        return {"status": "unhealthy", "service": "Firestore", "error": str(e)}
