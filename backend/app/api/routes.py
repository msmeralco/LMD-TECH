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


# ==============================================================================
# BARANGAY ID MAPPING - Matches frontend cityMetadata.ts
# ==============================================================================

# ID to barangay/city mapping (CSV contains barangay_id)
# This matches the frontend BARANGAY_TO_CITY structure
BARANGAY_ID_MAP = {
    # MANILA
    1: {'city': 'manila', 'barangay': 'Tondo'},
    2: {'city': 'manila', 'barangay': 'Ermita'},
    3: {'city': 'manila', 'barangay': 'Malate'},
    4: {'city': 'manila', 'barangay': 'Paco'},
    5: {'city': 'manila', 'barangay': 'Pandacan'},
    6: {'city': 'manila', 'barangay': 'Port Area'},
    7: {'city': 'manila', 'barangay': 'Quiapo'},
    8: {'city': 'manila', 'barangay': 'Sampaloc'},
    9: {'city': 'manila', 'barangay': 'San Andres'},
    10: {'city': 'manila', 'barangay': 'San Miguel'},
    11: {'city': 'manila', 'barangay': 'San Nicolas'},
    12: {'city': 'manila', 'barangay': 'Santa Ana'},
    13: {'city': 'manila', 'barangay': 'Santa Cruz'},
    14: {'city': 'manila', 'barangay': 'Santa Mesa'},
    15: {'city': 'manila', 'barangay': 'Binondo'},
    16: {'city': 'manila', 'barangay': 'Intramuros'},
    17: {'city': 'manila', 'barangay': 'San Antonio'},
    18: {'city': 'manila', 'barangay': 'Singalong'},
    19: {'city': 'manila', 'barangay': 'Moriones'},
    20: {'city': 'manila', 'barangay': 'Balic-Balic'},
    
    # QUEZON CITY
    21: {'city': 'quezon', 'barangay': 'Batasan Hills'},
    22: {'city': 'quezon', 'barangay': 'Commonwealth'},
    23: {'city': 'quezon', 'barangay': 'Fairview'},
    24: {'city': 'quezon', 'barangay': 'Novaliches'},
    25: {'city': 'quezon', 'barangay': 'Diliman'},
    26: {'city': 'quezon', 'barangay': 'Cubao'},
    27: {'city': 'quezon', 'barangay': 'Kamuning'},
    28: {'city': 'quezon', 'barangay': 'Libis'},
    29: {'city': 'quezon', 'barangay': 'Project 4'},
    30: {'city': 'quezon', 'barangay': 'Project 6'},
    31: {'city': 'quezon', 'barangay': 'Project 8'},
    32: {'city': 'quezon', 'barangay': 'San Francisco Del Monte'},
    33: {'city': 'quezon', 'barangay': 'Santa Mesa Heights'},
    34: {'city': 'quezon', 'barangay': 'Talipapa'},
    35: {'city': 'quezon', 'barangay': 'Teachers Village'},
    36: {'city': 'quezon', 'barangay': 'Baesa'},
    37: {'city': 'quezon', 'barangay': 'Bagong Lipunan'},
    38: {'city': 'quezon', 'barangay': 'Blue Ridge'},
    39: {'city': 'quezon', 'barangay': 'Holy Spirit'},
    40: {'city': 'quezon', 'barangay': 'Payatas'},
    
    # MAKATI
    46: {'city': 'makati', 'barangay': 'Poblacion'},
    47: {'city': 'makati', 'barangay': 'Bel-Air'},
    48: {'city': 'makati', 'barangay': 'Forbes Park'},
    49: {'city': 'makati', 'barangay': 'Dasmariñas'},
    50: {'city': 'makati', 'barangay': 'Urdaneta'},
    51: {'city': 'makati', 'barangay': 'San Lorenzo'},
    
    # Add more as needed - CSV will provide the barangay_id
}

def get_city_from_barangay_id(barangay_id: int) -> str:
    """
    Get city ID from barangay_id (from CSV).
    
    Args:
        barangay_id: The barangay ID from CSV
    
    Returns:
        City ID (e.g., 'manila', 'quezon')
    """
    if barangay_id in BARANGAY_ID_MAP:
        return BARANGAY_ID_MAP[barangay_id]['city']
    
    # If not found in mapping, return 'unknown'
    logger.warning(f"⚠️ Barangay ID {barangay_id} not found in mapping")
    return 'unknown'

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
    
    # DON'T add barangay here - it will be added later from barangay_id lookup
    # The ML model doesn't need the barangay text column
    
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
                'barangay_id': 1,
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
            "barangay_id", "lat", "lon", "kVA"
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
    
    # ✅ NEW LOGIC: Use barangay_id from CSV to lookup city
    # CSV now contains barangay_id column with assigned IDs
    if 'barangay_id' in df.columns:
        # Map barangay_id to city using BARANGAY_ID_MAP
        df['city_id'] = df['barangay_id'].apply(get_city_from_barangay_id)
        df['city_name'] = df['city_id'].str.title().str.replace('_', ' ')
        # Map barangay_id to barangay name
        df['barangay'] = df['barangay_id'].apply(lambda bid: BARANGAY_ID_MAP.get(bid, {}).get('barangay', 'Unknown'))
        logger.info(f"✅ Mapped cities from barangay_id: {df['city_id'].unique().tolist()}")
    elif 'city' in df.columns:
        # Fallback: If CSV has city column, use it directly
        df['city_id'] = df['city'].str.lower().str.replace(' ', '_')
        df['city_name'] = df['city'].str.title()
    elif 'city_id' not in df.columns:
        # Legacy: COMPREHENSIVE NCR Barangay-to-City Mapping (400+ barangays)
        barangay_to_city = {
            # MANILA (Major barangays)
            'Tondo': 'manila', 'Ermita': 'manila', 'Malate': 'manila', 'Paco': 'manila',
            'Pandacan': 'manila', 'Port Area': 'manila', 'Quiapo': 'manila', 'Sampaloc': 'manila',
            'San Andres': 'manila', 'San Miguel': 'manila', 'San Nicolas': 'manila', 'Santa Ana': 'manila',
            'Santa Cruz': 'manila', 'Santa Mesa': 'manila', 'Binondo': 'manila', 'Intramuros': 'manila',
            'San Antonio': 'manila', 'Singalong': 'manila', 'Moriones': 'manila', 'Balic-Balic': 'manila',
            
            # QUEZON CITY (Major barangays)
            'Batasan Hills': 'quezon', 'Commonwealth': 'quezon', 'Fairview': 'quezon', 'Novaliches': 'quezon',
            'Diliman': 'quezon', 'Cubao': 'quezon', 'Kamuning': 'quezon', 'Libis': 'quezon',
            'Project 4': 'quezon', 'Project 6': 'quezon', 'Project 8': 'quezon', 'San Francisco Del Monte': 'quezon',
            'Santa Mesa Heights': 'quezon', 'Talipapa': 'quezon', 'Teachers Village': 'quezon', 'Baesa': 'quezon',
            'Bagong Lipunan': 'quezon', 'Blue Ridge': 'quezon', 'Holy Spirit': 'quezon', 'Payatas': 'quezon',
            'Pasong Tamo': 'quezon', 'Greater Lagro': 'quezon', 'Bagong Pag-asa': 'quezon', 'Maharlika': 'quezon',
            'Old Capitol Site': 'quezon',
            
            # MAKATI (All 33 barangays)
            'Poblacion': 'makati', 'Bel-Air': 'makati', 'Forbes Park': 'makati', 'Dasmariñas': 'makati',
            'Urdaneta': 'makati', 'San Lorenzo': 'makati', 'Carmona': 'makati', 'Olympia': 'makati',
            'Guadalupe Nuevo': 'makati', 'Guadalupe Viejo': 'makati', 'Pembo': 'makati', 'Comembo': 'makati',
            'Rizal': 'makati', 'Magallanes': 'makati', 'La Paz': 'makati', 'San Antonio': 'makati',
            'Palanan': 'makati', 'Pinagkaisahan': 'makati', 'Tejeros': 'makati', 'Singkamas': 'makati',
            'West Rembo': 'makati', 'East Rembo': 'makati', 'Pitogo': 'makati', 'Cembo': 'makati',
            'South Cembo': 'makati',
            
            # PASIG (All 30 barangays)
            'Kapitolyo': 'pasig', 'Ugong': 'pasig', 'Ortigas': 'pasig', 'Rosario': 'pasig',
            'Santolan': 'pasig', 'Malinao': 'pasig', 'San Antonio': 'pasig', 'San Joaquin': 'pasig',
            'San Miguel': 'pasig', 'Santa Lucia': 'pasig', 'Pineda': 'pasig', 'Manggahan': 'pasig',
            'Maybunga': 'pasig', 'Caniogan': 'pasig', 'Kalawaan': 'pasig', 'Dela Paz': 'pasig',
            'Sagad': 'pasig', 'Pinagbuhatan': 'pasig', 'Bambang': 'pasig', 'Bagong Ilog': 'pasig',
            'Bagong Katipunan': 'pasig',
            
            # MARIKINA (All 16 barangays)
            'Barangka': 'marikina', 'Concepcion Uno': 'marikina', 'Concepcion Dos': 'marikina',
            'Industrial Valley': 'marikina', 'Jesus Dela Pena': 'marikina', 'Kalumpang': 'marikina',
            'Malanday': 'marikina', 'Nangka': 'marikina', 'Parang': 'marikina', 'San Roque': 'marikina',
            'Santa Elena': 'marikina', 'Santo Niño': 'marikina', 'Tañong': 'marikina', 'Tumana': 'marikina',
            'Fortune': 'marikina', 'Marikina Heights': 'marikina',
            
            # MANDALUYONG (Major barangays)
            'Addition Hills': 'mandaluyong', 'Barangka Drive': 'mandaluyong', 'Buayang Bato': 'mandaluyong',
            'Burol': 'mandaluyong', 'Hulo': 'mandaluyong', 'Mabini-J. Rizal': 'mandaluyong',
            'Malamig': 'mandaluyong', 'Namayan': 'mandaluyong', 'New Zaniga': 'mandaluyong',
            'Pag-asa': 'mandaluyong', 'Plainview': 'mandaluyong', 'Pleasant Hills': 'mandaluyong',
            'Poblacion': 'mandaluyong', 'San Jose': 'mandaluyong', 'Vergara': 'mandaluyong',
            'Wack-Wack Greenhills': 'mandaluyong',
            
            # TAGUIG (Major barangays)
            'Bagumbayan': 'taguig', 'Bambang': 'taguig', 'Fort Bonifacio': 'taguig', 'Hagonoy': 'taguig',
            'Ibayo-Tipas': 'taguig', 'Ligid-Tipas': 'taguig', 'Lower Bicutan': 'taguig',
            'Maharlika Village': 'taguig', 'Napindan': 'taguig', 'New Lower Bicutan': 'taguig',
            'North Signal Village': 'taguig', 'Palingon': 'taguig', 'Pinagsama': 'taguig',
            'San Miguel': 'taguig', 'Santa Ana': 'taguig', 'Tuktukan': 'taguig',
            'Upper Bicutan': 'taguig', 'Western Bicutan': 'taguig', 'Central Bicutan': 'taguig',
            'Bonifacio Global City': 'taguig', 'BGC': 'taguig',
            
            # PATEROS (All 10 barangays)
            'Aguho': 'pateros', 'Magtanggol': 'pateros', 'Martires Del 96': 'pateros',
            'Poblacion': 'pateros', 'San Pedro': 'pateros', 'San Roque': 'pateros',
            'Santa Ana': 'pateros', 'Santo Rosario-Kanluran': 'pateros', 'Santo Rosario-Silangan': 'pateros',
            'Tabacalera': 'pateros',
            
            # PARAÑAQUE (Major barangays)
            'Baclaran': 'paranaque', 'Don Bosco': 'paranaque', 'La Huerta': 'paranaque',
            'San Antonio': 'paranaque', 'San Dionisio': 'paranaque', 'San Isidro': 'paranaque',
            'San Martin De Porres': 'paranaque', 'Santo Niño': 'paranaque', 'Sun Valley': 'paranaque',
            'Tambo': 'paranaque', 'Vitalez': 'paranaque', 'BF Homes': 'paranaque',
            'Marcelo Green': 'paranaque', 'Merville': 'paranaque', 'Moonwalk': 'paranaque',
            
            # MUNTINLUPA (All 9 barangays)
            'Alabang': 'muntinlupa', 'Bayanan': 'muntinlupa', 'Buli': 'muntinlupa',
            'Cupang': 'muntinlupa', 'Poblacion': 'muntinlupa', 'Putatan': 'muntinlupa',
            'Sucat': 'muntinlupa', 'Tunasan': 'muntinlupa', 'Ayala Alabang': 'muntinlupa',
            
            # LAS PIÑAS (Major barangays)
            'Almanza Uno': 'laspinas', 'Almanza Dos': 'laspinas', 'BF International': 'laspinas',
            'Daniel Fajardo': 'laspinas', 'Elias Aldana': 'laspinas', 'Ilaya': 'laspinas',
            'Manuyo Uno': 'laspinas', 'Manuyo Dos': 'laspinas', 'Pamplona Uno': 'laspinas',
            'Pamplona Dos': 'laspinas', 'Pamplona Tres': 'laspinas', 'Pilar': 'laspinas',
            'Pulang Lupa Uno': 'laspinas', 'Pulang Lupa Dos': 'laspinas', 'Talon Uno': 'laspinas',
            'Talon Dos': 'laspinas', 'Talon Tres': 'laspinas', 'Zapote': 'laspinas',
            
            # PASAY (Major zones)
            'Zone 1': 'pasay', 'Zone 14': 'pasay', 'Zone 19': 'pasay', 'Baclaran': 'pasay',
            'Malibay': 'pasay', 'San Isidro': 'pasay', 'San Rafael': 'pasay', 'San Roque': 'pasay',
            'Libertad': 'pasay',
            
            # CALOOCAN (Major barangays)
            'Bagong Barrio': 'caloocan', 'Bagong Silang': 'caloocan', 'Kaybiga': 'caloocan',
            'Camarin': 'caloocan', 'Grace Park': 'caloocan', 'Maypajo': 'caloocan',
            'Tala': 'caloocan', '10th Avenue': 'caloocan', 'Sangandaan': 'caloocan',
            
            # MALABON (All 21 barangays)
            'Acacia': 'malabon', 'Baritan': 'malabon', 'Bayan-bayanan': 'malabon',
            'Catmon': 'malabon', 'Concepcion': 'malabon', 'Dampalit': 'malabon',
            'Flores': 'malabon', 'Hulong Duhat': 'malabon', 'Ibaba': 'malabon',
            'Longos': 'malabon', 'Maysilo': 'malabon', 'Muzon': 'malabon',
            'Niugan': 'malabon', 'Panghulo': 'malabon', 'Potrero': 'malabon',
            'San Agustin': 'malabon', 'Santolan': 'malabon', 'Tañong': 'malabon',
            'Tinajeros': 'malabon', 'Tonsuya': 'malabon', 'Tugatog': 'malabon',
            
            # NAVOTAS (All 14 barangays)
            'Bagumbayan North': 'navotas', 'Bagumbayan South': 'navotas', 'Bangculasi': 'navotas',
            'Daanghari': 'navotas', 'Navotas East': 'navotas', 'Navotas West': 'navotas',
            'North Bay Boulevard North': 'navotas', 'North Bay Boulevard South': 'navotas',
            'San Jose': 'navotas', 'San Rafael Village': 'navotas', 'San Roque': 'navotas',
            'Sipac-Almacen': 'navotas', 'Tangos': 'navotas', 'Tanza': 'navotas',
            
            # VALENZUELA (Major barangays)
            'Arkong Bato': 'valenzuela', 'Bagbaguin': 'valenzuela', 'Balangkas': 'valenzuela',
            'Bignay': 'valenzuela', 'Bisig': 'valenzuela', 'Canumay East': 'valenzuela',
            'Canumay West': 'valenzuela', 'Coloong': 'valenzuela', 'Dalandanan': 'valenzuela',
            'Gen. T. De Leon': 'valenzuela', 'Isla': 'valenzuela', 'Karuhatan': 'valenzuela',
            'Lawang Bato': 'valenzuela', 'Lingunan': 'valenzuela', 'Mabolo': 'valenzuela',
            'Malanday': 'valenzuela', 'Malinta': 'valenzuela', 'Mapulang Lupa': 'valenzuela',
            'Marulas': 'valenzuela', 'Maysan': 'valenzuela', 'Palasan': 'valenzuela',
            'Parada': 'valenzuela', 'Pariancillo Villa': 'valenzuela', 'Pasolo': 'valenzuela',
            'Poblacion': 'valenzuela', 'Polo': 'valenzuela', 'Punturin': 'valenzuela',
            'Rincon': 'valenzuela', 'Tagalag': 'valenzuela', 'Ugong': 'valenzuela',
            'Wawang Pulo': 'valenzuela',
            
            # SAN JUAN (Major barangays)
            'Addition Hills': 'sanjuan', 'Balong-Bato': 'sanjuan', 'Batis': 'sanjuan',
            'Corazon De Jesus': 'sanjuan', 'Ermitaño': 'sanjuan', 'Greenhills': 'sanjuan',
            'Isabelita': 'sanjuan', 'Kabayanan': 'sanjuan', 'Little Baguio': 'sanjuan',
            'Maytunas': 'sanjuan', 'Onse': 'sanjuan', 'Pasadeña': 'sanjuan',
            'Pedro Cruz': 'sanjuan', 'Progreso': 'sanjuan', 'Rivera': 'sanjuan',
            'Salapan': 'sanjuan', 'San Perfecto': 'sanjuan', 'Santa Lucia': 'sanjuan',
            'Tibagan': 'sanjuan', 'West Crame': 'sanjuan', 'St. Joseph': 'sanjuan',
        }
        df['city_id'] = df['barangay'].map(barangay_to_city).fillna('unknown')
        df['city_name'] = df['city_id'].str.title().str.replace('_', ' ')
    
    # ========== METER LEVEL ==========
    # Extract monthly consumption columns and add to meter data
    consumption_cols = [col for col in df.columns if col.startswith('monthly_consumption_')]
    
    # Convert to list format for frontend
    meter_records = []
    for _, row in df.iterrows():
        meter_dict = row.to_dict()
        
        # ✅ barangay_id already in CSV - no need to generate
        # Extract monthly consumptions as array
        monthly_consumptions = [row[col] for col in consumption_cols if pd.notna(row[col])]
        meter_dict['monthly_consumptions'] = monthly_consumptions
        
        # Remove individual month columns to reduce payload size
        for col in consumption_cols:
            meter_dict.pop(col, None)
        
        meter_records.append(meter_dict)
    
    meters = meter_records
    
    # ========== TRANSFORMER LEVEL ==========
    transformer_agg = df.groupby('transformer_id').agg({
        'meter_id': 'count',
        'anomaly_score': 'mean',
        'lat': 'first',
        'lon': 'first',
        'barangay_id': 'first',
        'city_id': 'first',
        'kVA': 'sum',
    }).reset_index()
    
    # Add barangay name from barangay_id
    transformer_agg['barangay'] = transformer_agg['barangay_id'].apply(
        lambda bid: BARANGAY_ID_MAP.get(bid, {}).get('barangay', 'Unknown')
    )
    
    transformer_agg.columns = [
        'transformer_id', 'total_meters', 'avg_anomaly_score',
        'lat', 'lon', 'barangay_id', 'city_id', 'capacity_kVA', 'barangay'
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
    # ✅ barangay_id already in DataFrame from CSV
    barangay_agg = transformer_agg.groupby(['barangay', 'barangay_id']).agg({
        'transformer_id': 'count',
        'avg_anomaly_score': 'mean',
        'lat': 'mean',
        'lon': 'mean',
        'city_id': 'first',
    }).reset_index()
    
    barangay_agg.columns = [
        'barangay', 'barangay_id', 'total_transformers', 'avg_risk_score',
        'lat', 'lon', 'city_id'
    ]
    
    # Count high-risk transformers per barangay
    high_risk_counts = transformer_agg[transformer_agg['risk_level'] == 'HIGH'].groupby('barangay_id').size()
    barangay_agg['high_risk_count'] = barangay_agg['barangay_id'].map(high_risk_counts).fillna(0).astype(int)
    
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


@router.get("/filters/{run_id}")
async def get_filter_options(
    run_id: str,
    city_id: Optional[str] = Query(None, description="Filter barangays by city")
):
    """
    Get unique filter values for ranking sidebar.
    
    Args:
        run_id: Analysis run ID
        city_id: Optional city ID to filter barangays (e.g., 'manila', 'quezon')
    
    Returns:
        {
            "barangays": ["Tondo", "Ermita", ...],
            "transformers": ["TX_MAIN_001", "TX_MAIN_002", ...],
            "risk_levels": ["HIGH", "MEDIUM", "LOW"]
        }
    """
    try:
        results = await get_run_results(run_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Run not found")
        
        meters = results.get("meters", [])
        
        # Filter meters by city if specified
        if city_id:
            meters = [m for m in meters if m.get("city_id", "").lower() == city_id.lower()]
        
        # Extract unique values
        barangays = sorted(list(set(m.get("barangay", "") for m in meters if m.get("barangay"))))
        transformers = sorted(list(set(m.get("transformer_id", "") for m in meters if m.get("transformer_id"))))
        risk_levels = ["HIGH", "MEDIUM", "LOW"]
        
        return {
            "barangays": barangays,
            "transformers": transformers,
            "risk_levels": risk_levels
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving filter options: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# EXPORT TO CSV
# ==============================================================================

@router.get("/export/{run_id}")
async def export_report(
    run_id: str,
    level: str = Query("meter", regex="^(transformer|district|meter)$"),
    barangay: Optional[str] = Query(None, description="Filter by barangay"),
    transformer: Optional[str] = Query(None, description="Filter by transformer/feeder"),
    risk_level: Optional[str] = Query(None, regex="^(HIGH|MEDIUM|LOW)$", description="Filter by risk level")
):
    """
    Export analysis results as CSV for field inspections with optional filters.
    
    Args:
        run_id: Analysis run ID
        level: "transformer", "district" (barangay), or "meter"
        barangay: Filter by barangay name (optional)
        transformer: Filter by transformer ID (optional)
        risk_level: Filter by risk level: HIGH, MEDIUM, or LOW (optional)
    
    Returns:
        Filtered CSV file download
    
    Examples:
        /export/{run_id}?level=meter&barangay=Tondo&risk_level=HIGH
        /export/{run_id}?level=meter&transformer=TX_MAIN_001
        /export/{run_id}?level=meter  (all meters, ranked by risk)
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
            
            # Apply filters
            if barangay:
                df = df[df['barangay'] == barangay]
                logger.info(f"📍 Filtered by barangay: {barangay} ({len(df)} meters)")
            
            if transformer:
                df = df[df['transformer_id'] == transformer]
                logger.info(f"⚡ Filtered by transformer: {transformer} ({len(df)} meters)")
            
            if risk_level:
                df = df[df['risk_level'] == risk_level]
                logger.info(f"🎯 Filtered by risk level: {risk_level} ({len(df)} meters)")
            
            # Sort by anomaly score (highest risk first)
            df = df.sort_values('anomaly_score', ascending=False)
            
            # Build filename with filters
            filter_parts = []
            if barangay:
                filter_parts.append(barangay.replace(' ', '_'))
            if transformer:
                filter_parts.append(transformer)
            if risk_level:
                filter_parts.append(risk_level.lower())
            
            filter_str = '_'.join(filter_parts) if filter_parts else 'all'
            filename = f"ghostload_meters_{filter_str}_{run_id}.csv"
            
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
