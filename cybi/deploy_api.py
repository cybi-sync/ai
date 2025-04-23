from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import io
from uuid import uuid4
import asyncio
import time

from cybi.data_module import CYBIDataProcessor
from cybi.main import CYBISystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cybi_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CYBI Smartshoe AI API",
    description="API for weight prediction and health monitoring from CYBI smartshoe data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize CYBI system
cybi_system = None

# Model cache
models = {
    'weight_model': None,
    'health_model': None,
    'is_loaded': False
}

# Define data models
class SensorData(BaseModel):
    """Model for incoming sensor data from the smartshoe."""
    timestamp: float
    pressure_sensors: List[float] = Field(..., description="Readings from pressure sensors")
    accelerometer: Dict[str, List[float]] = Field(..., description="Accelerometer readings (x, y, z) for left and right shoe")
    gyroscope: Dict[str, List[float]] = Field(..., description="Gyroscope readings (x, y, z) for left and right shoe")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Any additional sensor data")

class PredictionRequest(BaseModel):
    """Model for batch prediction requests."""
    data: List[SensorData]

class PredictionResult(BaseModel):
    """Model for prediction results."""
    request_id: str
    timestamp: float
    weight_prediction: float
    health_prediction: Dict[str, Any]
    confidence: Dict[str, float]

class PredictionResponse(BaseModel):
    """Model for the overall prediction response."""
    request_id: str
    results: List[PredictionResult]
    processing_time: float
    status: str

class StatusResponse(BaseModel):
    """Model for API status responses."""
    status: str
    version: str
    models_loaded: bool
    uptime: float
    timestamp: float

# Track API start time
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Load models and initialize system on startup."""
    global cybi_system
    try:
        logger.info("Initializing CYBI system")
        cybi_system = CYBISystem()
        
        # Load models in background
        asyncio.create_task(load_models_async())
        
        logger.info("CYBI API startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

async def load_models_async():
    """Load models asynchronously to avoid blocking startup."""
    global models
    try:
        logger.info("Loading models in background")
        
        # Load weight prediction model
        weight_model_path = './models/weight_prediction_model'
        if os.path.exists(weight_model_path):
            models['weight_model'] = tf.keras.models.load_model(weight_model_path)
        else:
            logger.warning(f"Weight model not found at {weight_model_path}")
        
        # Check for ensemble health models
        ensemble_model_paths = [f for f in os.listdir('./models') if f.startswith('health_prediction_ensemble_model_')]
        
        if ensemble_model_paths:
            health_models = []
            for path in sorted(ensemble_model_paths):
                health_models.append(tf.keras.models.load_model(f'./models/{path}'))
            models['health_model'] = health_models
        else:
            # Load single health prediction model
            health_model_path = './models/health_prediction_model'
            if os.path.exists(health_model_path):
                models['health_model'] = tf.keras.models.load_model(health_model_path)
            else:
                logger.warning(f"Health model not found at {health_model_path}")
        
        # Mark models as loaded if at least one model is available
        if models['weight_model'] is not None or models['health_model'] is not None:
            models['is_loaded'] = True
            logger.info("Models loaded successfully")
        else:
            logger.error("No models could be loaded")
    
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        models['is_loaded'] = False

def sensor_data_to_dataframe(data_list: List[SensorData]) -> pd.DataFrame:
    """Convert sensor data from API format to pandas DataFrame for processing."""
    rows = []
    
    for data in data_list:
        row = {}
        
        # Add timestamp
        row['timestamp'] = data.timestamp
        
        # Add pressure sensors
        for i, value in enumerate(data.pressure_sensors):
            side = 'left' if i < 6 else 'right'
            position = ['heel', 'arch', 'mid', 'ball', 'toe', 'edge'][i % 6]
            row[f'pressure_{side}_{position}'] = value
        
        # Add accelerometer data
        for side in ['left', 'right']:
            if side in data.accelerometer:
                for i, axis in enumerate(['x', 'y', 'z']):
                    if i < len(data.accelerometer[side]):
                        row[f'accel_{side}_{axis}'] = data.accelerometer[side][i]
        
        # Add gyroscope data
        for side in ['left', 'right']:
            if side in data.gyroscope:
                for i, axis in enumerate(['x', 'y', 'z']):
                    if i < len(data.gyroscope[side]):
                        row[f'gyro_{side}_{axis}'] = data.gyroscope[side][i]
        
        # Add any additional data
        if data.additional_data:
            for key, value in data.additional_data.items():
                if isinstance(value, (int, float, str, bool)):
                    row[key] = value
        
        rows.append(row)
    
    return pd.DataFrame(rows)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get API status and information."""
    return {
        "status": "operational",
        "version": "1.0.0",
        "models_loaded": models['is_loaded'],
        "uptime": time.time() - start_time,
        "timestamp": time.time()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(prediction_request: PredictionRequest):
    """
    Make predictions using the CYBI models.
    
    - **prediction_request**: Batch of sensor data from the smartshoe
    
    Returns a prediction response with weight and health predictions
    """
    global cybi_system, models
    
    # Check if models are loaded
    if not models['is_loaded']:
        raise HTTPException(status_code=503, detail="Models are not loaded yet. Please try again later.")
    
    # Generate request ID
    request_id = str(uuid4())
    start_time = time.time()
    
    try:
        # Convert input data to DataFrame
        df = sensor_data_to_dataframe(prediction_request.data)
        
        # Check if data is empty
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid data provided in the request.")
        
        # Make predictions
        predictions = cybi_system.predict(df, models=models)
        
        # Format results
        results = []
        for i, (index, row) in enumerate(predictions.iterrows()):
            # Extract prediction details
            weight = float(row['weight_prediction'])
            
            # Extract health prediction
            health_prediction = {"class": int(row['health_class'])}
            
            # Extract confidence values
            confidence = {}
            
            # Add probabilities for each health class if available
            health_prob_columns = [col for col in row.index if col.startswith('health_prob_class_')]
            if health_prob_columns:
                class_probs = {}
                for col in health_prob_columns:
                    class_idx = int(col.split('_')[-1])
                    class_probs[str(class_idx)] = float(row[col])
                health_prediction["probabilities"] = class_probs
                
                # Overall confidence is the probability of the predicted class
                if str(health_prediction["class"]) in class_probs:
                    confidence["health"] = class_probs[str(health_prediction["class"])]
                else:
                    confidence["health"] = 0.0
            else:
                # Binary classification
                confidence["health"] = float(row.get('health_prob', 0.0))
            
            # Add weight confidence (inverse of error estimate)
            confidence["weight"] = 0.95  # Placeholder - in a real system this would be model-based
            
            # Create result object
            result = PredictionResult(
                request_id=request_id,
                timestamp=prediction_request.data[min(i, len(prediction_request.data)-1)].timestamp,
                weight_prediction=weight,
                health_prediction=health_prediction,
                confidence=confidence
            )
            
            results.append(result)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log prediction
        logger.info(f"Processed prediction request {request_id} with {len(results)} results in {processing_time:.4f}s")
        
        return PredictionResponse(
            request_id=request_id,
            results=results,
            processing_time=processing_time,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")

@app.post("/upload-data")
async def upload_sensor_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a CSV file with sensor data and process it in the background.
    
    - **file**: CSV file with sensor data
    
    Returns a job ID for tracking the processing
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Generate job ID
    job_id = str(uuid4())
    
    try:
        # Read the file content
        contents = await file.read()
        
        # Save file to disk for processing
        file_path = f"./data/uploaded_{job_id}.csv"
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Schedule background processing
        background_tasks.add_task(process_uploaded_file, file_path, job_id)
        
        return JSONResponse(content={
            "job_id": job_id,
            "status": "processing",
            "message": "File uploaded successfully and processing started"
        })
    
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

async def process_uploaded_file(file_path: str, job_id: str):
    """Process an uploaded file in the background."""
    global cybi_system, models
    
    try:
        logger.info(f"Processing uploaded file for job {job_id}")
        
        # Read data
        df = pd.read_csv(file_path)
        
        # Make predictions
        predictions = cybi_system.predict(df, models=models)
        
        # Save results
        output_path = f"./results/processed_{job_id}.csv"
        predictions.to_csv(output_path, index=False)
        
        logger.info(f"Processing complete for job {job_id}, results saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a background processing job.
    
    - **job_id**: Job ID returned from the upload endpoint
    
    Returns the job status and result location if complete
    """
    # Check if results file exists
    results_path = f"./results/processed_{job_id}.csv"
    if os.path.exists(results_path):
        return JSONResponse(content={
            "job_id": job_id,
            "status": "complete",
            "results_path": results_path
        })
    
    # Check if input file exists (still processing)
    input_path = f"./data/uploaded_{job_id}.csv"
    if os.path.exists(input_path):
        return JSONResponse(content={
            "job_id": job_id,
            "status": "processing"
        })
    
    # Neither file exists
    raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")

@app.get("/train")
async def trigger_training(background_tasks: BackgroundTasks, generate_data: bool = False):
    """
    Trigger model training or retraining.
    
    - **generate_data**: Whether to generate synthetic data for training
    
    Returns a training job ID
    """
    global cybi_system
    
    # Generate job ID
    job_id = str(uuid4())
    
    # Schedule background training
    background_tasks.add_task(run_training, job_id, generate_data)
    
    return JSONResponse(content={
        "job_id": job_id,
        "status": "training_started",
        "message": "Model training has been started in the background"
    })

async def run_training(job_id: str, generate_data: bool):
    """Run model training in the background."""
    global cybi_system, models
    
    try:
        logger.info(f"Starting model training for job {job_id}")
        
        # Run the pipeline
        results = cybi_system.run_pipeline(generate_data=generate_data)
        
        # Update loaded models
        await load_models_async()
        
        # Save training results
        with open(f"./results/training_{job_id}.json", "w") as f:
            # Convert complex objects to JSON-serializable format
            serializable_results = {
                "weight_metrics": results["weight_metrics"],
                "health_metrics": results["health_metrics"],
                "timestamp": datetime.now().isoformat(),
                "job_id": job_id
            }
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Training complete for job {job_id}")
    
    except Exception as e:
        logger.error(f"Error during training job {job_id}: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("cybi.deploy_api:app", host="0.0.0.0", port=8000, reload=False) 