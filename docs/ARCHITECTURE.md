# System Architecture - People Tracking System

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Design](#component-design)
4. [Data Flow](#data-flow)
5. [Module Structure](#module-structure)
6. [Design Patterns](#design-patterns)
7. [Technology Stack](#technology-stack)

---

## Overview

The People Tracking System is a comprehensive multi-object tracking (MOT) system designed for real-time and batch processing of video sequences. The system integrates deep learning-based detection, state-of-the-art tracking algorithms, and production-ready deployment infrastructure.

### Key Characteristics
- **Modular Design**: Loosely coupled components with clear interfaces
- **Extensible**: Easy to add new tracking algorithms and features
- **Production-Ready**: Includes API, web UI, and Docker deployment
- **Research-Oriented**: Includes novel Transformer-based tracker

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Web UI (Streamlit)  │  REST API (FastAPI)  │  CLI Tools   │
└──────────────────────┬───────────────────────┬─────────────┘
                       │                       │
                       ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Inference Layer                          │
├─────────────────────────────────────────────────────────────┤
│  Video Processor  │  DeepSORT Tracker  │  Result Exporter  │
└────────────────────┬───────────────────────┬───────────────┘
                     │                       │
                     ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                               │
├─────────────────────────────────────────────────────────────┤
│  Detection (YOLOv8)  │  ReID Model  │  Transformer Tracker │
└──────────────────────┬───────────────────────┬──────────────┘
                       │                       │
                       ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Algorithm Layer                      │
├─────────────────────────────────────────────────────────────┤
│  Kalman Filter  │  Hungarian Algorithm  │  Feature Extract │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
Input Video/Images
    │
    ▼
┌─────────────────┐
│  Video Loader   │ → Frame Extraction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLOv8         │ → Person Detection (bboxes + confidences)
│  Detector       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │ → Appearance Features (ReID or Color Histogram)
│  Extractor      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DeepSORT       │ → Track Association & State Management
│  Tracker        │
└────────┬────────┘
         │
         ├──► Kalman Filter (Motion Prediction)
         ├──► Hungarian Algorithm (Optimal Matching)
         └──► Track State Machine (Tentative/Confirmed/Deleted)
         │
         ▼
┌─────────────────┐
│  Visualizer     │ → Annotated Video with Track IDs
└────────┬────────┘
         │
         ▼
    Output Video/JSON
```

---

## Component Design

### 1. Detection Module (`src/models/detection/`)

**Purpose**: Detect people in video frames

**Components**:
- `yolo_detector.py`: YOLOv8 integration wrapper

**Key Features**:
- Configurable confidence threshold
- Non-maximum suppression (NMS)
- Batch processing support
- GPU acceleration

**Interface**:
```python
class YOLODetector:
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect people in frame
        Returns: List of [x1, y1, x2, y2, confidence]
        """
```

---

### 2. Tracking Module (`src/models/tracking/`)

**Purpose**: Track detected objects across frames

#### 2.1 DeepSORT Tracker (`deepsort.py`)

**Core Components**:

1. **KalmanFilter Class**
   - State: `[x, y, a, h, vx, vy, va, vh]`
   - 8-dimensional state vector
   - Constant velocity motion model
   - Methods: `predict()`, `update()`, `project()`, `gating_distance()`

2. **Track Class**
   - Manages individual track lifecycle
   - States: Tentative → Confirmed → Deleted
   - Feature gallery management (deque with max size)
   - Age and time_since_update tracking

3. **DeepSORT Class**
   - Main tracking orchestrator
   - Cascade matching strategy
   - IoU-based fallback matching
   - Appearance feature matching
   - Track initialization and deletion

**Key Algorithms**:
- **Cascade Matching**: Prioritize matching younger tracks
- **IoU Matching**: Geometric overlap for occluded cases
- **Appearance Matching**: Cosine distance on feature vectors
- **Hungarian Algorithm**: Optimal assignment (via scipy)

#### 2.2 Transformer Tracker (`transformer_tracker.py`)

**Architecture**:
- **Encoder**: Processes detection features with self-attention
- **Decoder**: Processes track features with cross-attention to detections
- **Output Heads**:
  - Match score head (detection-track association)
  - Bbox refinement head (boundary box correction)
  - Track state head (active/inactive/new prediction)

**Innovation**: End-to-end learnable association

#### 2.3 Multi-Camera Tracker (`multi_camera_tracker.py`)

**Features**:
- Global ID management across cameras
- Homography-based world coordinate projection
- Camera topology graph
- Cross-camera re-identification

---

### 3. ReID Module (`src/models/reid/`)

**Purpose**: Extract discriminative appearance features

**Architecture**:
- **Backbone**: ResNet50 (pre-trained on ImageNet)
- **Attention**: Channel Attention mechanism
- **Feature Dimension**: 2048
- **Loss Functions**:
  - Triplet Loss (metric learning)
  - Cross-Entropy Loss (classification)
  - Combined Loss (weighted sum)

**Training**:
- Dataset: MOT20 person crops
- Hard negative mining
- Batch sampling strategy

---

### 4. Inference Module (`src/inference/`)

**Purpose**: End-to-end video processing pipeline

**Components**:
- `deepsort_tracker.py`: `DeepSORTVideoTracker` class
- `main.py`: CLI interface

**Features**:
- Video file processing
- Image sequence processing (MOT20 format)
- Webcam streaming
- Real-time visualization
- Result export (JSON + video)

---

### 5. Training Module (`src/training/`)

**Purpose**: Train detection and ReID models

**Components**:
- `train_detector.py`: YOLOv8 fine-tuning on MOT20
- `train_reid.py`: ReID model training
- `train_transformer_tracker.py`: Transformer tracker training

**Features**:
- Weights & Biases integration
- Checkpoint management
- Validation metrics
- Learning rate scheduling

---

### 6. Evaluation Module (`src/evaluation/`)

**Purpose**: Calculate MOT metrics

**Components**:
- `mot_metrics.py`: `MOTEvaluator` class

**Metrics**:
- MOTA (Multiple Object Tracking Accuracy)
- MOTP (Multiple Object Tracking Precision)
- IDF1 (ID F1 Score)
- Precision, Recall
- ID Switches, Fragments

---

### 7. API Module (`src/api/`)

**Purpose**: RESTful API for system access

**Components**:
- `main.py`: Basic FastAPI server
- `production_api.py`: Production API with Celery + Redis

**Endpoints**:
- `POST /upload`: Upload video
- `POST /track`: Submit tracking job
- `GET /status/{job_id}`: Check job status
- `GET /result/{job_id}`: Get results
- `DELETE /job/{job_id}`: Delete job

---

### 8. UI Module (`src/ui/`)

**Purpose**: Web-based user interface

**Components**:
- `app.py`: Streamlit application
- `main.py`: UI launcher

**Features**:
- Video upload
- Real-time processing
- Results visualization
- Statistics dashboard

---

### 9. Applications Module (`src/applications/`)

**Purpose**: Domain-specific applications

**Components**:
- `retail_analytics.py`: Retail store analytics
- `traffic_monitoring.py`: Traffic flow monitoring

**Features**:
- Zone-based counting
- Dwell time analysis
- Anomaly detection
- Heat map generation

---

### 10. Analysis Module (`src/models/analysis/`)

**Purpose**: Advanced trajectory and crowd analysis

**Components**:
- `trajectory_analysis.py`: Trajectory prediction and anomaly detection
- `crowd_density.py`: Density estimation and hotspot detection

---

## Data Flow

### Tracking Pipeline

```
Frame Input
    │
    ├─► Detection (YOLOv8)
    │   └─► [bbox, confidence] × N
    │
    ├─► Feature Extraction
    │   ├─► ReID Model (if available)
    │   └─► Color Histogram (fallback)
    │   └─► Feature Vector × N
    │
    ├─► Track Prediction (Kalman Filter)
    │   └─► Predicted States × M
    │
    ├─► Association (DeepSORT)
    │   ├─► Cascade Matching
    │   ├─► IoU Matching
    │   └─► Hungarian Algorithm
    │   └─► Matches: (track_idx, detection_idx)
    │
    ├─► Track Update
    │   ├─► Update matched tracks (Kalman + features)
    │   ├─► Create new tracks (unmatched detections)
    │   └─► Delete old tracks (max_age exceeded)
    │
    └─► Output
        ├─► Annotated Frame
        └─► Tracking Data (JSON)
```

### Training Pipeline

```
Dataset (MOT20)
    │
    ├─► Data Loader
    │   ├─► MOT20Dataset (detection training)
    │   └─► MOT20ReIDDataset (ReID training)
    │
    ├─► Model Forward
    │   ├─► Detection: YOLOv8
    │   └─► ReID: ResNet50 + Attention
    │
    ├─► Loss Calculation
    │   ├─► Detection: YOLO loss
    │   └─► ReID: Triplet + Cross-Entropy
    │
    ├─► Backward Pass
    │   └─► Gradient Update
    │
    └─► Checkpoint Save
```

---

## Module Structure

```
src/
├── api/                    # REST API server
│   ├── main.py            # Basic API
│   └── production_api.py  # Production API (Celery + Redis)
│
├── inference/              # Video processing
│   ├── main.py            # CLI interface
│   └── deepsort_tracker.py # DeepSORT integration
│
├── models/
│   ├── detection/         # Detection models
│   │   └── yolo_detector.py
│   │
│   ├── tracking/          # Tracking algorithms
│   │   ├── deepsort.py   # DeepSORT implementation
│   │   ├── transformer_tracker.py # Transformer tracker
│   │   ├── multi_camera_tracker.py # Multi-camera
│   │   └── simple_tracker.py # Simple baseline
│   │
│   ├── reid/              # Re-identification
│   │   └── reid_model.py
│   │
│   └── analysis/          # Analysis tools
│       ├── trajectory_analysis.py
│       └── crowd_density.py
│
├── training/              # Training scripts
│   ├── train_detector.py
│   ├── train_reid.py
│   └── train_transformer_tracker.py
│
├── evaluation/            # Evaluation metrics
│   └── mot_metrics.py
│
├── data/                  # Data loaders
│   ├── mot_dataset.py
│   └── reid_dataset.py
│
├── applications/          # Domain applications
│   ├── retail_analytics.py
│   └── traffic_monitoring.py
│
├── ui/                    # Web interface
│   └── app.py
│
└── utils/                 # Utilities
    └── mot_utils.py
```

---

## Design Patterns

### 1. Strategy Pattern
- **Tracking Algorithms**: DeepSORT, Transformer, Simple Tracker
- **Feature Extractors**: ReID Model, Color Histogram
- **Matching Strategies**: Cascade, IoU, Appearance

### 2. State Machine Pattern
- **Track States**: Tentative → Confirmed → Deleted
- **State Transitions**: Based on consecutive detections and age

### 3. Factory Pattern
- **Model Loading**: Dynamic model instantiation based on config
- **Tracker Creation**: Factory method for different tracker types

### 4. Observer Pattern
- **Progress Tracking**: Callbacks for training progress
- **Event Handling**: API job status updates

### 5. Pipeline Pattern
- **Video Processing**: Sequential stages (detect → extract → track → visualize)
- **Training Pipeline**: Data → Model → Loss → Optimizer

---

## Technology Stack

### Core Libraries
- **PyTorch**: Deep learning framework
- **Ultralytics YOLOv8**: Object detection
- **OpenCV**: Video processing and visualization
- **NumPy**: Numerical computations
- **SciPy**: Optimization algorithms (Hungarian)

### Web & API
- **FastAPI**: REST API framework
- **Streamlit**: Web UI framework
- **Uvicorn**: ASGI server
- **Celery**: Background task processing
- **Redis**: Job queue and caching

### Data & Evaluation
- **motmetrics**: MOT evaluation metrics
- **Pillow**: Image processing
- **Albumentations**: Data augmentation

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Reverse proxy (production)

### Development
- **pytest**: Testing framework
- **black**: Code formatting
- **Weights & Biases**: Experiment tracking

---

## Configuration Management

### Configuration Files

```
configs/
├── tracking_config.yaml      # Tracking parameters
├── detection_training.yaml   # Detection training config
├── reid_training.yaml        # ReID training config
├── transformer_tracker_training.yaml # Transformer config
└── deployment_config.yaml    # Deployment settings
```

### Key Parameters

**Tracking**:
- `max_dist`: Maximum appearance distance (0.32)
- `max_age`: Maximum frames without detection (50)
- `n_init`: Frames before track confirmation (5)
- `min_confidence`: Minimum detection confidence (0.15)

**Detection**:
- `conf_threshold`: Detection confidence threshold (0.15)
- `iou_threshold`: NMS IoU threshold (0.45)

---

## Performance Considerations

### Optimization Strategies
1. **GPU Acceleration**: CUDA support for detection and ReID
2. **Batch Processing**: Process multiple frames when possible
3. **Feature Caching**: Reuse features for similar detections
4. **Model Quantization**: ONNX export for faster inference
5. **TensorRT**: GPU-optimized inference (optional)

### Scalability
- **Horizontal Scaling**: Multiple API workers
- **Background Jobs**: Celery workers for async processing
- **Caching**: Redis for frequently accessed data
- **Load Balancing**: Nginx for API distribution

---

## Security Considerations

### API Security
- Input validation for uploaded files
- File size limits
- Rate limiting (production)
- Authentication (optional, for production)

### Data Privacy
- No persistent storage of video data (optional)
- Secure file handling
- Temporary file cleanup

---

## Extension Points

### Adding New Trackers
1. Implement tracker interface
2. Add to `src/models/tracking/`
3. Update factory method
4. Add configuration options

### Adding New Features
1. Create feature extractor
2. Integrate with tracking pipeline
3. Update configuration
4. Add evaluation metrics

### Adding New Applications
1. Create application module in `src/applications/`
2. Implement domain-specific logic
3. Add API endpoints (if needed)
4. Create UI components (if needed)

---

*Document Version: 1.0*  
*Last Updated: 2025-11-22*  
*System Version: Final Optimized*

