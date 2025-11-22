"""Production API server with background job processing"""
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import uuid
from pathlib import Path
import asyncio
import json
import os

# Optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis not available. Job queue will use in-memory storage.")

try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    print("Warning: celery not available. Background tasks will run synchronously.")

app = FastAPI(
    title="People Tracking System API",
    description="Production-grade multi-object tracking API",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for job queue (with fallback)
if REDIS_AVAILABLE:
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        redis_client.ping()
    except Exception:
        redis_client = None
        print("Warning: Redis connection failed. Using in-memory job storage.")
else:
    redis_client = None

# In-memory job storage (fallback)
job_storage = {}

# Celery for background tasks (optional)
if CELERY_AVAILABLE and REDIS_AVAILABLE:
    try:
        celery_app = Celery('tracking_tasks', broker='redis://localhost:6379/0')
    except Exception:
        celery_app = None
        print("Warning: Celery initialization failed.")
else:
    celery_app = None

# Global tracker instance
tracker = None


class TrackingRequest(BaseModel):
    """Request model for tracking"""
    video_url: Optional[str] = None
    confidence_threshold: float = 0.25
    max_age: int = 30
    n_init: int = 3
    visualization: bool = True


class TrackingResponse(BaseModel):
    """Response model for tracking"""
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Job status model"""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0-100
    result_url: Optional[str] = None
    error: Optional[str] = None


def _get_job_storage():
    """Get job storage (Redis or in-memory)"""
    if redis_client:
        return redis_client
    return job_storage


def _set_job_status(job_id: str, status: str, **kwargs):
    """Set job status"""
    storage = _get_job_storage()
    if isinstance(storage, dict):
        if job_id not in storage:
            storage[job_id] = {}
        storage[job_id]['status'] = status
        storage[job_id].update(kwargs)
    else:
        # Redis
        storage.hset(f'job:{job_id}', 'status', status)
        for key, value in kwargs.items():
            storage.hset(f'job:{job_id}', key, str(value))


def _get_job_status(job_id: str) -> Dict:
    """Get job status"""
    storage = _get_job_storage()
    if isinstance(storage, dict):
        return storage.get(job_id, {})
    else:
        # Redis
        if not storage.exists(f'job:{job_id}'):
            return {}
        data = storage.hgetall(f'job:{job_id}')
        return {k.decode('utf-8') if isinstance(k, bytes) else k: 
                v.decode('utf-8') if isinstance(v, bytes) else v 
                for k, v in data.items()}


def process_video_sync(job_id: str, video_path: str, config: Dict):
    """Synchronous video processing (fallback when Celery unavailable)"""
    try:
        _set_job_status(job_id, 'processing', progress=0)
        
        # Process video
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f'{job_id}_tracked.mp4'
        results_path = output_dir / f'{job_id}_results.json'
        
        # Run tracker
        from src.inference.deepsort_tracker import DeepSORTVideoTracker
        
        tracker_config = {
            'detection': {'conf_threshold': config.get('confidence_threshold', 0.25)},
            'tracking': {
                'max_dist': 0.2,
                'max_iou_distance': 0.7,
                'max_age': config.get('max_age', 30),
                'n_init': config.get('n_init', 3)
            }
        }
        
        video_tracker = DeepSORTVideoTracker(
            detection_model_path='models/checkpoints/yolov8n.pt',
            reid_model_path=None,  # Can be configured
            config=tracker_config
        )
        
        results = video_tracker.process_video(
            video_path=video_path,
            output_path=str(output_path),
            visualize=config.get('visualization', True)
        )
        
        # Save results
        video_tracker.save_results(results, str(results_path))
        
        # Update status
        _set_job_status(job_id, 'completed', progress=100, result_url=str(output_path))
        
        return {'status': 'completed', 'result_url': str(output_path)}
    
    except Exception as e:
        _set_job_status(job_id, 'failed', error=str(e))
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize tracker on startup"""
    global tracker
    
    try:
        from src.inference.deepsort_tracker import DeepSORTVideoTracker
        
        detection_model = 'models/checkpoints/yolov8n.pt'
        if not Path(detection_model).exists():
            print(f"Warning: Detection model not found at {detection_model}")
            print("API will work but tracking may fail. Please train or download models.")
            return
        
        tracker = DeepSORTVideoTracker(
            detection_model_path=detection_model,
            reid_model_path=None,  # Optional
            config={
                'detection': {'conf_threshold': 0.25},
                'tracking': {
                    'max_dist': 0.2,
                    'max_iou_distance': 0.7,
                    'max_age': 30,
                    'n_init': 3
                }
            }
        )
        
        print("Tracker initialized successfully")
    except Exception as e:
        print(f"Warning: Tracker initialization failed: {e}")
        print("API will start but tracking endpoints may not work.")


if CELERY_AVAILABLE and celery_app:
    @celery_app.task(bind=True)
    def process_video_task(self, job_id, video_path, config):
        """Background task for video processing"""
        return process_video_sync(job_id, video_path, config)


@app.post("/api/v2/track", response_model=TrackingResponse)
async def track_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = 0.25,
    max_age: int = 30,
    visualization: bool = True
):
    """
    Upload video for tracking
    Returns job ID for status checking
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_dir = Path('uploads')
    upload_dir.mkdir(exist_ok=True)
    
    video_path = upload_dir / f'{job_id}_{file.filename}'
    
    with open(video_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    # Create job entry
    _set_job_status(job_id, 'pending', video_path=str(video_path))
    
    # Submit background task
    config = {
        'confidence_threshold': confidence_threshold,
        'max_age': max_age,
        'visualization': visualization
    }
    
    if CELERY_AVAILABLE and celery_app:
        # Use Celery for async processing
        task = process_video_task.apply_async(args=[job_id, str(video_path), config])
    else:
        # Use FastAPI background tasks
        background_tasks.add_task(process_video_sync, job_id, str(video_path), config)
    
    return TrackingResponse(
        job_id=job_id,
        status="pending",
        message="Video uploaded successfully. Processing started."
    )


@app.get("/api/v2/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get tracking job status"""
    job_data = _get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job_data.get('status', 'unknown')
    result_url = job_data.get('result_url')
    error = job_data.get('error')
    
    # Get progress
    progress = float(job_data.get('progress', 0))
    if status == 'processing':
        progress = progress if progress > 0 else 50  # Placeholder
    elif status == 'completed':
        progress = 100.0
    
    return JobStatus(
        job_id=job_id,
        status=status,
        progress=progress,
        result_url=result_url,
        error=error
    )


@app.get("/api/v2/result/{job_id}")
async def get_result_video(job_id: str):
    """Download tracked video result"""
    job_data = _get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job_data.get('status', '')
    
    if status != 'completed':
        raise HTTPException(status_code=400, detail=f"Job status: {status}")
    
    result_url = job_data.get('result_url')
    if not result_url:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result_path = Path(result_url)
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        result_path,
        media_type='video/mp4',
        filename=f'tracked_{job_id}.mp4'
    )


@app.get("/api/v2/metrics/{job_id}")
async def get_tracking_metrics(job_id: str):
    """Get tracking metrics/statistics"""
    results_path = Path('outputs') / f'{job_id}_results.json'
    
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return JSONResponse(content=results.get('statistics', {}))


@app.delete("/api/v2/job/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated files"""
    job_data = _get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get file paths
    video_path = job_data.get('video_path')
    result_url = job_data.get('result_url')
    
    # Delete files
    if video_path:
        Path(video_path).unlink(missing_ok=True)
    if result_url:
        Path(result_url).unlink(missing_ok=True)
    
    results_path = Path('outputs') / f'{job_id}_results.json'
    results_path.unlink(missing_ok=True)
    
    # Delete job from storage
    storage = _get_job_storage()
    if isinstance(storage, dict):
        storage.pop(job_id, None)
    else:
        storage.delete(f'job:{job_id}')
    
    return {"message": "Job deleted successfully"}


@app.get("/api/v2/health")
async def health_check():
    """Health check endpoint"""
    redis_connected = False
    if redis_client:
        try:
            redis_client.ping()
            redis_connected = True
        except Exception:
            pass
    
    return {
        "status": "healthy",
        "tracker_loaded": tracker is not None,
        "redis_connected": redis_connected,
        "celery_available": CELERY_AVAILABLE and celery_app is not None
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "People Tracking System API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/v2/health"
    }


if __name__ == '__main__':
    import uvicorn
    
    # Create necessary directories
    Path('uploads').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)  # Single worker for development


