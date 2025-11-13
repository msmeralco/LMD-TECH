"""
API Routes for GhostLoad Mapper
================================

FastAPI endpoints that connect frontend → backend → ML pipeline → Firestore.

Complete workflow:
1. Frontend uploads CSV
2. Backend validates and passes to ML pipeline
3. ML pipeline returns predictions
4. Backend aggregates data (city → barangay → transformer → meter)
5. Backend saves to Firestore
6. Frontend fetches and displays on map

Author: Backend Team
Date: November 13, 2025
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, JSONResponse
import pandas as pd
from io import StringIO
import uuid
from datetime import datetime
import time
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
import json

# Import Firestore helpers
from app.db.firestore import (
    save_run_results,
    get_run_results,
    list_recent_runs,
    delete_run,
    save_transformer_metadata,
    health_check as firestore_health_check
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL FIX: Add PROJECT ROOT to Python path (not machine_learning folder)
# This allows: "from machine_learning.pipeline.inference_pipeline import ..."
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    logger.info(f"📂 Added project root to sys.path: {project_root}")

# Import ML pipeline
try:
    from machine_learning.pipeline.inference_pipeline import InferencePipeline, predict_batch_from_dataframe
    logger.info("✅ ML pipeline imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import ML pipeline: {e}")
    logger.error(f"   Project root: {project_root}")
    logger.error(f"   Sys.path: {sys.path[:3]}")
    # Create mock pipeline for development
    class InferencePipeline:
        def predict(self, data):
            logger.warning("⚠️ Using mock ML pipeline - predictions will be empty")
            return []
    predict_batch_from_dataframe = None

# Initialize router
router = APIRouter(prefix="/api", tags=["main"])

# Initialize ML pipeline (singleton)
ml_pipeline = None

def validate_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and prepare DataFrame for ML pipeline.
    Ensures all required fields are present and properly formatted.
    
    CRITICAL: Handles missing consumption months by padding with zeros
    to match the model's expected 12-month window.
    
    Args:
        df: Raw DataFrame from CSV upload
    
    Returns:
        Cleaned DataFrame ready for ML pipeline
    """
    df_clean = df.copy()
    
    # Ensure required columns exist with proper types
    if 'meter_id' in df_clean.columns:
        df_clean['meter_id'] = df_clean['meter_id'].astype(str)
    
    if 'transformer_id' in df_clean.columns:
        df_clean['transformer_id'] = df_clean['transformer_id'].astype(str)
    
    # Ensure numeric columns are float (add if missing)
    numeric_cols = ['lat', 'lon', 'kVA']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        else:
            # Add missing columns with default values
            logger.warning(f"⚠️ Column '{col}' missing from CSV, adding defaults")
            df_clean[col] = 0.0
    
    # Ensure consumption columns are numeric
    consumption_cols = [col for col in df_clean.columns if col.startswith('monthly_consumption_')]
    for col in consumption_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # CRITICAL FIX: Pad consumption columns to 12 months if needed
    # Model was trained with 12 months, CSV might have less
    if len(consumption_cols) < 12:
        logger.warning(f"⚠️ CSV has only {len(consumption_cols)} consumption months, model expects 12")
        logger.info("🔧 Padding with forward-fill to reach 12 months")
        
        # Get existing months from column names
        existing_months = []
        for col in consumption_cols:
            try:
                month_str = col.replace('monthly_consumption_', '')
                existing_months.append(int(month_str))
            except ValueError:
                continue
        
        if existing_months:
            existing_months.sort()
            last_month = existing_months[-1]
            
            # Add missing months by forward-filling the last month's data
            months_needed = 12 - len(consumption_cols)
            
            for i in range(1, months_needed + 1):
                # Increment month (simple YYYYMM increment)
                year = last_month // 100
                month = last_month % 100
                month += 1
                if month > 12:
                    month = 1
                    year += 1
                new_month = year * 100 + month
                
                new_col = f'monthly_consumption_{new_month}'
                logger.info(f"   Adding {new_col} (forward-filled)")
                
                # Forward-fill: copy last available month's consumption
                last_col = f'monthly_consumption_{last_month}'
                if last_col in df_clean.columns:
                    df_clean[new_col] = df_clean[last_col]
                else:
                    df_clean[new_col] = 0.0
                
                last_month = new_month
    
    # Add missing optional fields with defaults
    if 'customer_class' not in df_clean.columns:
        df_clean['customer_class'] = 'residential'
    
    if 'barangay' not in df_clean.columns:
        df_clean['barangay'] = 'Unknown'
    
    logger.info(f"✅ Data prepared: {len(df_clean)} rows, "
               f"{len([c for c in df_clean.columns if c.startswith('monthly_consumption_')])} consumption months")
    
    return df_clean

def get_ml_pipeline():
    """Get or initialize ML pipeline (lazy loading)"""
    global ml_pipeline
    if ml_pipeline is None:
        try:
            logger.info("🔄 Initializing ML pipeline...")
            ml_pipeline = InferencePipeline()
            logger.info("✅ ML pipeline initialized successfully")
            
            # Test prediction to verify output format
            test_data = {
                'meter_id': 'TEST_INIT',
                'transformer_id': 'TX_TEST',
                'customer_class': 'residential',
                'barangay': 'Test',
                'lat': 14.5,
                'lon': 120.9,
                'monthly_consumption_202411': 100,
                'monthly_consumption_202412': 110,
                'monthly_consumption_202501': 105,
                'monthly_consumption_202502': 95,
                'monthly_consumption_202503': 115,
                'monthly_consumption_202504': 120,
                'monthly_consumption_202505': 125,
                'monthly_consumption_202506': 118,
                'monthly_consumption_202507': 112,
                'monthly_consumption_202508': 108,
                'monthly_consumption_202509': 122,
                'monthly_consumption_202510': 130,
                'kVA': 50
            }
            
            test_result = ml_pipeline.predict(test_data)
            test_dict = test_result.to_dict()
            
            logger.info(f"🧪 Test prediction keys: {list(test_dict.keys())}")
            logger.info(f"🧪 ML output format verified: risk_level={test_dict.get('risk_level')}, "
                       f"anomaly_score={test_dict.get('anomaly_score'):.3f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML pipeline: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"ML pipeline initialization failed: {str(e)}"
            )
    return ml_pipeline


# ==============================================================================
# MAIN ENDPOINT: Upload CSV + Run ML Pipeline
# ==============================================================================

@router.post("/run")
async def run_analysis(file: UploadFile = File(...)):
    """
    🎯 MAIN ENDPOINT: Upload CSV and run complete analysis pipeline.
    
    This endpoint:
    1. Receives meter consumption CSV
    2. Validates required columns
    3. Runs ML pipeline to get predictions
    4. Aggregates data hierarchically (city → barangay → transformer → meter)
    5. Saves everything to Firestore
    6. Returns run_id for frontend to fetch results
    
    Expected CSV columns:
    - meter_id, transformer_id, customer_class, barangay, lat, lon, kVA
    - monthly_consumption_YYYYMM (at least 6 months)
    
    Returns:
        {
            "run_id": "abc-123-def",
            "status": "completed",
            "total_meters": 1500,
            "total_transformers": 120,
            "high_risk_count": 75,
            "processing_time_seconds": 12.5
        }
    """
    start_time = time.time()
    
    try:
        # Step 1: Read and validate CSV
        logger.info("📥 Reading uploaded CSV...")
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode()))
        
        logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Validate required columns
        required_cols = {
            "meter_id", "transformer_id", "customer_class", 
            "barangay", "lat", "lon", "kVA"
        }
        
        # Check for monthly consumption columns
        consumption_cols = [col for col in df.columns if col.startswith("monthly_consumption_")]
        
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {list(missing)}"
            )
        
        if len(consumption_cols) < 6:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 6 monthly consumption columns. Found: {len(consumption_cols)}"
            )
        
        logger.info(f"✅ Validation passed ({len(consumption_cols)} consumption months)")
        
        # Clean and prepare data
        logger.info("🧹 Preparing data for ML pipeline...")
        df_clean = validate_and_prepare_dataframe(df)
        logger.info(f"✅ Data prepared: {len(df_clean)} rows ready for ML")
        
        # Step 2: Run ML pipeline
        logger.info("🤖 Running ML pipeline...")
        pipeline = get_ml_pipeline()
        
        # Get predictions for all meters
        predictions = pipeline.predict(df_clean)
        
        # Convert predictions to dictionaries
        prediction_dicts = [p.to_dict() for p in predictions]
        
        logger.info(f"✅ ML pipeline completed ({len(prediction_dicts)} predictions)")
        
        # DEBUG: Log first prediction to verify structure
        if prediction_dicts:
            logger.info(f"🔍 Sample prediction keys: {list(prediction_dicts[0].keys())}")
            logger.info(f"🔍 Sample prediction: {prediction_dicts[0]}")
        
        # Step 3: Merge predictions with original data
        logger.info("🔗 Merging predictions with meter data...")
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame(prediction_dicts)
        logger.info(f"🔍 Prediction DataFrame columns: {pred_df.columns.tolist()}")
        
        # ADAPTER LAYER: Ensure all required columns exist
        # This makes the backend resilient to ML output variations
        required_pred_cols = ['meter_id', 'risk_level', 'anomaly_score', 'confidence', 'explanation']
        
        # Check what's missing and add defaults if needed
        for col in required_pred_cols:
            if col not in pred_df.columns:
                logger.warning(f"⚠️ Column '{col}' missing from ML output, adding defaults")
                
                if col == 'meter_id':
                    # This should never be missing
                    raise ValueError("ML pipeline must return 'meter_id' field")
                elif col == 'risk_level':
                    # Derive from anomaly_score if available
                    if 'anomaly_score' in pred_df.columns:
                        pred_df['risk_level'] = pred_df['anomaly_score'].apply(
                            lambda x: "HIGH" if x >= 0.7 else "MEDIUM" if x >= 0.4 else "LOW"
                        )
                    else:
                        pred_df['risk_level'] = "UNKNOWN"
                elif col == 'anomaly_score':
                    pred_df['anomaly_score'] = 0.5  # Neutral score
                elif col == 'confidence':
                    # Derive from anomaly_score if available
                    if 'anomaly_score' in pred_df.columns:
                        pred_df['confidence'] = (pred_df['anomaly_score'] - 0.5).abs() * 2
                    else:
                        pred_df['confidence'] = 0.5
                elif col == 'explanation':
                    pred_df['explanation'] = "Anomaly detection completed"
        
        # Merge with original data
        df_merged = df.merge(
            pred_df[required_pred_cols],
            on='meter_id',
            how='left'
        )
        
        logger.info(f"✅ Merged data: {len(df_merged)} rows, columns: {df_merged.columns.tolist()[:10]}...")
        
        # Step 4: Aggregate data hierarchically
        logger.info("📊 Aggregating data...")
        aggregated_data = aggregate_hierarchical_data(df_merged)
        
        # Step 5: Generate run_id and save to Firestore
        run_id = str(uuid.uuid4())
        processing_time = time.time() - start_time
        
        aggregated_data["run_id"] = run_id
        aggregated_data["processing_time"] = processing_time
        aggregated_data["model_version"] = "1.0"
        
        logger.info("💾 Saving to Firestore...")
        await save_run_results(run_id, aggregated_data)
        
        logger.info(f"✅ Analysis complete! Run ID: {run_id}")
        
        # Return summary
        return {
            "run_id": run_id,
            "status": "completed",
            "total_meters": aggregated_data["total_meters"],
            "total_transformers": aggregated_data["total_transformers"],
            "total_barangays": aggregated_data["total_barangays"],
            "total_cities": aggregated_data["total_cities"],
            "high_risk_count": aggregated_data["high_risk_count"],
            "processing_time_seconds": round(processing_time, 2)
        }
        
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def aggregate_hierarchical_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate meter predictions into hierarchical structure for frontend.
    
    Creates:
    - City-level aggregations
    - Barangay-level aggregations
    - Transformer-level aggregations
    - Meter-level data
    - High-risk summary
    
    Args:
        df: DataFrame with meter data + ML predictions
    
    Returns:
        Dictionary with hierarchical data
    """
    
    # Ensure city_id exists (extract from barangay if needed)
    if 'city_id' not in df.columns:
        # Map barangay to city (you can customize this mapping)
        barangay_to_city = {
            'Tondo': 'manila',
            'Ermita': 'manila',
            'San Nicolas': 'manila',
            'Sampaloc': 'manila',
            'Pandacan': 'manila',
            # Add more mappings as needed
        }
        df['city_id'] = df['barangay'].map(barangay_to_city).fillna('unknown')
        df['city_name'] = df['city_id'].str.title()
    
    # ========== METER LEVEL ==========
    meters = df.to_dict(orient='records')
    
    # ========== TRANSFORMER LEVEL ==========
    transformer_agg = df.groupby('transformer_id').agg({
        'meter_id': 'count',
        'anomaly_score': 'mean',
        'lat': 'first',
        'lon': 'first',
        'barangay': 'first',
        'city_id': 'first',
        'kVA': 'sum',
    }).reset_index()
    
    transformer_agg.columns = [
        'transformer_id', 'total_meters', 'avg_anomaly_score',
        'lat', 'lon', 'barangay', 'city_id', 'capacity_kVA'
    ]
    
    # Count suspicious meters per transformer
    suspicious_counts = df[df['risk_level'] == 'HIGH'].groupby('transformer_id').size()
    transformer_agg['suspicious_meter_count'] = transformer_agg['transformer_id'].map(suspicious_counts).fillna(0).astype(int)
    
    # Calculate risk level for transformer
    transformer_agg['risk_level'] = transformer_agg['avg_anomaly_score'].apply(
        lambda x: "HIGH" if x >= 0.7 else "MEDIUM" if x >= 0.4 else "LOW"
    )
    
    # Calculate consumption statistics
    consumption_cols = [col for col in df.columns if col.startswith('monthly_consumption_')]
    transformer_agg['avg_consumption'] = df.groupby('transformer_id')[consumption_cols].mean().mean(axis=1).values
    transformer_agg['median_consumption'] = df.groupby('transformer_id')[consumption_cols].median().median(axis=1).values
    
    transformers = transformer_agg.to_dict(orient='records')
    
    # ========== BARANGAY LEVEL ==========
    barangay_agg = transformer_agg.groupby('barangay').agg({
        'transformer_id': 'count',
        'avg_anomaly_score': 'mean',
        'lat': 'mean',
        'lon': 'mean',
        'city_id': 'first',
    }).reset_index()
    
    barangay_agg.columns = [
        'barangay', 'total_transformers', 'avg_risk_score',
        'lat', 'lon', 'city_id'
    ]
    
    # Count high-risk transformers per barangay
    high_risk_counts = transformer_agg[transformer_agg['risk_level'] == 'HIGH'].groupby('barangay').size()
    barangay_agg['high_risk_count'] = barangay_agg['barangay'].map(high_risk_counts).fillna(0).astype(int)
    
    barangays = barangay_agg.to_dict(orient='records')
    
    # ========== CITY LEVEL ==========
    city_agg = barangay_agg.groupby('city_id').agg({
        'total_transformers': 'sum',
        'avg_risk_score': 'mean',
        'lat': 'mean',
        'lon': 'mean',
    }).reset_index()
    
    city_agg.columns = [
        'city_id', 'total_transformers', 'avg_risk_score', 'lat', 'lon'
    ]
    
    # Count high-risk transformers per city
    city_high_risk = barangay_agg.groupby('city_id')['high_risk_count'].sum()
    city_agg['high_risk_count'] = city_agg['city_id'].map(city_high_risk).fillna(0).astype(int)
    
    # Add city names
    city_agg['city_name'] = city_agg['city_id'].str.title()
    
    cities = city_agg.to_dict(orient='records')
    
    # ========== HIGH-RISK SUMMARY ==========
    top_transformers = transformer_agg.nlargest(10, 'avg_anomaly_score')[
        ['transformer_id', 'barangay', 'avg_anomaly_score', 'suspicious_meter_count', 'risk_level']
    ].to_dict(orient='records')
    
    high_risk_summary = {
        "total_high_risk_meters": len(df[df['risk_level'] == 'HIGH']),
        "total_high_risk_transformers": len(transformer_agg[transformer_agg['risk_level'] == 'HIGH']),
        "most_anomalous_city": city_agg.loc[city_agg['avg_risk_score'].idxmax(), 'city_name'] if len(city_agg) > 0 else "N/A",
        "most_anomalous_barangay": barangay_agg.loc[barangay_agg['avg_risk_score'].idxmax(), 'barangay'] if len(barangay_agg) > 0 else "N/A",
        "most_anomalous_transformer": top_transformers[0]['transformer_id'] if top_transformers else "N/A",
        "top_10_transformers": top_transformers
    }
    
    return {
        "total_meters": len(df),
        "total_transformers": len(transformer_agg),
        "total_barangays": len(barangay_agg),
        "total_cities": len(city_agg),
        "high_risk_count": high_risk_summary["total_high_risk_meters"],
        "cities": cities,
        "barangays": barangays,
        "transformers": transformers,
        "meters": meters,
        "high_risk_summary": high_risk_summary
    }


# ==============================================================================
# GET RESULTS
# ==============================================================================

@router.get("/results/{run_id}")
async def get_results(run_id: str):
    """
    Retrieve complete analysis results for a specific run.
    
    Frontend uses this to:
    - Display cities on initial map load
    - Zoom to barangay when city clicked
    - Show transformers when barangay clicked
    - Display meter details in popup
    - Populate "High-Risk Locations" sidebar
    
    Returns:
        Complete hierarchical data structure
    """
    try:
        results = await get_run_results(run_id)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Run {run_id} not found"
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# EXPORT TO CSV
# ==============================================================================

@router.get("/export/{run_id}")
async def export_report(
    run_id: str,
    level: str = Query("transformer", regex="^(transformer|district|meter)$")
):
    """
    Export analysis results as CSV for field inspections.
    
    Args:
        run_id: Analysis run ID
        level: "transformer", "district" (barangay), or "meter"
    
    Returns:
        CSV file download
    """
    try:
        results = await get_run_results(run_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Generate CSV based on level
        if level == "transformer":
            df = pd.DataFrame(results["transformers"])
            filename = f"ghostload_transformers_{run_id}.csv"
            
        elif level == "district":
            df = pd.DataFrame(results["barangays"])
            filename = f"ghostload_barangays_{run_id}.csv"
            
        elif level == "meter":
            df = pd.DataFrame(results["meters"])
            # Select relevant columns for export
            export_cols = [
                'meter_id', 'transformer_id', 'barangay', 'lat', 'lon',
                'customer_class', 'kVA', 'risk_level', 'anomaly_score',
                'confidence', 'explanation'
            ]
            df = df[[col for col in export_cols if col in df.columns]]
            filename = f"ghostload_meters_{run_id}.csv"
        
        csv_data = df.to_csv(index=False)
        
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# GEOJSON FOR MAP
# ==============================================================================

@router.get("/transformers.geojson")
async def get_transformers_geojson(run_id: Optional[str] = None):
    """
    Get transformer data in GeoJSON format for Leaflet map.
    
    Args:
        run_id: Optional run ID. If not provided, returns latest run.
    
    Returns:
        GeoJSON FeatureCollection
    """
    try:
        # Get results
        if run_id:
            results = await get_run_results(run_id)
        else:
            # Get latest run
            recent_runs = await list_recent_runs(limit=1)
            if not recent_runs:
                return {
                    "type": "FeatureCollection",
                    "features": []
                }
            results = await get_run_results(recent_runs[0]["run_id"])
        
        if not results:
            return {
                "type": "FeatureCollection",
                "features": []
            }
        
        # Convert transformers to GeoJSON
        features = []
        for transformer in results["transformers"]:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [transformer["lon"], transformer["lat"]]
                },
                "properties": {
                    "transformer_id": transformer["transformer_id"],
                    "barangay": transformer["barangay"],
                    "total_meters": transformer["total_meters"],
                    "suspicious_meter_count": transformer["suspicious_meter_count"],
                    "risk_level": transformer["risk_level"],
                    "avg_anomaly_score": transformer["avg_anomaly_score"],
                }
            })
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
        
    except Exception as e:
        logger.error(f"❌ GeoJSON generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# UTILITY ENDPOINTS
# ==============================================================================

@router.get("/runs")
async def list_runs(limit: int = Query(10, ge=1, le=100)):
    """
    List recent analysis runs.
    
    Args:
        limit: Number of runs to return (1-100)
    
    Returns:
        List of run summaries
    """
    try:
        runs = await list_recent_runs(limit=limit)
        return {"runs": runs}
        
    except Exception as e:
        logger.error(f"❌ Failed to list runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/runs/{run_id}")
async def delete_analysis_run(run_id: str):
    """
    Delete an analysis run and its results.
    
    Args:
        run_id: Run ID to delete
    
    Returns:
        Status message
    """
    try:
        result = await delete_run(run_id)
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to delete run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Checks:
    - API is running
    - ML pipeline is loaded
    - Firestore is accessible
    
    Returns:
        Health status
    """
    try:
        # Check ML pipeline
        ml_status = "healthy" if ml_pipeline is not None else "not_initialized"
        
        # Check Firestore
        firestore_status = await firestore_health_check()
        
        return {
            "status": "healthy",
            "api": "running",
            "ml_pipeline": ml_status,
            "firestore": firestore_status["status"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
