# Function Reference - People Tracking System

## Table of Contents
1. [Core Tracking Functions](#core-tracking-functions)
2. [Detection Functions](#detection-functions)
3. [Feature Extraction Functions](#feature-extraction-functions)
4. [Matching Functions](#matching-functions)
5. [Utility Functions](#utility-functions)
6. [API Functions](#api-functions)
7. [Training Functions](#training-functions)

---

## Core Tracking Functions

### `DeepSORT.update(detections, frame)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Main tracking update function

**Parameters**:
- `detections` (np.ndarray): Array of detections `[x1, y1, x2, y2, confidence]`
- `frame` (np.ndarray): Current frame image (BGR format)

**Returns**:
- `List[List[float, float, float, float, int]]`: Active tracks `[x1, y1, x2, y2, track_id]`

**Algorithm**:
1. Filter detections by confidence
2. Extract appearance features
3. Predict track positions (Kalman Filter)
4. Match detections to tracks (cascade matching)
5. Update matched tracks
6. Create new tracks for unmatched detections
7. Delete old tracks

**Example**:
```python
tracker = DeepSORT(reid_model=reid_model)
tracks = tracker.update(detections, frame)
```

---

### `KalmanFilter.predict(mean, covariance)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Predict next state using motion model

**Parameters**:
- `mean` (np.ndarray): Current state mean `[8]`
- `covariance` (np.ndarray): Current state covariance `[8, 8]`

**Returns**:
- `mean` (np.ndarray): Predicted state mean
- `covariance` (np.ndarray): Predicted state covariance

**Mathematical Formulation**:
```
x_k|k-1 = F * x_k-1|k-1
P_k|k-1 = F * P_k-1|k-1 * F^T + Q
```

---

### `KalmanFilter.update(mean, covariance, measurement)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Update state estimate with measurement

**Parameters**:
- `mean` (np.ndarray): Predicted state mean
- `covariance` (np.ndarray): Predicted state covariance
- `measurement` (np.ndarray): Measurement `[cx, cy, a, h]`

**Returns**:
- `mean` (np.ndarray): Updated state mean
- `covariance` (np.ndarray): Updated state covariance

**Mathematical Formulation**:
```
K = P_k|k-1 * H^T * (H * P_k|k-1 * H^T + R)^-1
x_k|k = x_k|k-1 + K * (z_k - H * x_k|k-1)
P_k|k = (I - K * H) * P_k|k-1
```

---

### `Track.update(measurement, feature)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Update track with new detection

**Parameters**:
- `measurement` (np.ndarray): Detection measurement `[cx, cy, a, h]`
- `feature` (np.ndarray, optional): Appearance feature vector

**Returns**: None (updates track state)

**Actions**:
1. Update Kalman Filter state
2. Append feature to gallery
3. Increment hit counter
4. Reset time_since_update
5. Promote to Confirmed if hits >= n_init

---

## Detection Functions

### `YOLODetector.detect(frame)`

**Location**: `src/models/detection/yolo_detector.py`

**Purpose**: Detect people in frame

**Parameters**:
- `frame` (np.ndarray): Input frame (BGR format)

**Returns**:
- `List[List[float]]`: Detections `[x1, y1, x2, y2, confidence]`

**Example**:
```python
detector = YOLODetector(model_path='yolov8n.pt')
detections = detector.detect(frame)
```

---

### `DeepSORTVideoTracker._detect_people(frame)`

**Location**: `src/inference/deepsort_tracker.py`

**Purpose**: Detect people using YOLOv8

**Parameters**:
- `frame` (np.ndarray): Input frame

**Returns**:
- `np.ndarray`: Detections `[N, 5]` where columns are `[x1, y1, x2, y2, conf]`

**Configuration**:
- `conf_threshold`: 0.15
- `iou_threshold`: 0.45 (NMS)

---

## Feature Extraction Functions

### `DeepSORT._extract_features(detections, frame)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Extract appearance features for detections

**Parameters**:
- `detections` (np.ndarray): Detection bounding boxes
- `frame` (np.ndarray): Current frame

**Returns**:
- `np.ndarray`: Feature vectors `[N, feature_dim]`

**Modes**:
1. **ReID Model** (if available): 2048-dim features
2. **Color Histogram** (fallback): 160-dim features

---

### `DeepSORT._extract_reid_features(detections, frame)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Extract features using ReID model

**Parameters**:
- `detections` (np.ndarray): Detection bounding boxes
- `frame` (np.ndarray): Current frame

**Returns**:
- `np.ndarray`: ReID features `[N, 2048]`

**Process**:
1. Crop person regions from frame
2. Resize to (256, 128)
3. Normalize (ImageNet stats)
4. Forward pass through ReID model
5. L2 normalize

---

### `DeepSORT._extract_color_features(detections, frame)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Extract color histogram features

**Parameters**:
- `detections` (np.ndarray): Detection bounding boxes
- `frame` (np.ndarray): Current frame

**Returns**:
- `np.ndarray`: Color features `[N, 160]`

**Process**:
1. Crop person regions
2. Split into top/bottom halves
3. Convert to HSV
4. Compute histograms (32 bins H/S, 16 bins V)
5. Weighted combination (0.6 top, 0.4 bottom)
6. L2 normalize

---

## Matching Functions

### `DeepSORT._cascade_match(detections, features, detection_boxes)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Two-pass cascade matching

**Parameters**:
- `detections` (np.ndarray): Detection measurements
- `features` (np.ndarray): Appearance features
- `detection_boxes` (np.ndarray): Detection bounding boxes

**Returns**:
- `matches` (List[Tuple[int, int]]): Matched pairs `(track_idx, det_idx)`
- `unmatched_tracks` (List[int]): Unmatched track indices
- `unmatched_detections` (List[int]): Unmatched detection indices

**Algorithm**:
1. First pass: Appearance + motion matching
2. Second pass: IoU matching for unmatched

---

### `DeepSORT._match(detections, features)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Match detections to tracks using appearance and motion

**Parameters**:
- `detections` (np.ndarray): Detection measurements
- `features` (np.ndarray): Appearance features

**Returns**:
- `matches`, `unmatched_tracks`, `unmatched_detections`

**Process**:
1. Compute cost matrix (appearance + motion)
2. Apply Hungarian algorithm
3. Filter by adaptive thresholds
4. Return matches

---

### `DeepSORT._compute_cost_matrix(detections, features)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Compute association cost matrix

**Parameters**:
- `detections` (np.ndarray): Detection measurements
- `features` (np.ndarray): Appearance features

**Returns**:
- `np.ndarray`: Cost matrix `[n_tracks, n_detections]`

**Cost Function**:
```
cost[i,j] = 0.2 * d_motion(i,j) + 0.8 * d_appearance(i,j)
```

Where:
- `d_motion`: Mahalanobis distance (gating)
- `d_appearance`: Cosine distance on features

---

### `DeepSORT._iou_match(tracks, detections)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Match using IoU for occluded cases

**Parameters**:
- `tracks` (List[Track]): Track objects
- `detections` (np.ndarray): Detection bounding boxes

**Returns**:
- `List[Tuple[int, int]]`: Matched pairs

**Algorithm**:
1. Compute IoU matrix
2. Hungarian algorithm (maximize IoU)
3. Filter by minimum IoU threshold (0.25)

---

### `DeepSORT._compute_iou(bbox1, bbox2)`

**Location**: `src/models/tracking/deepsort.py`

**Purpose**: Compute IoU between two bounding boxes

**Parameters**:
- `bbox1` (np.ndarray): `[x1, y1, x2, y2]`
- `bbox2` (np.ndarray): `[x1, y1, x2, y2]`

**Returns**:
- `float`: IoU value [0, 1]

**Formula**:
```
IoU = intersection_area / union_area
```

---

## Utility Functions

### `DeepSORTVideoTracker.process_video(video_path, output_path, visualize)`

**Location**: `src/inference/deepsort_tracker.py`

**Purpose**: Process entire video sequence

**Parameters**:
- `video_path` (str): Path to video, image sequence directory, or webcam index
- `output_path` (str, optional): Path to save output video
- `visualize` (bool): Whether to draw bounding boxes

**Returns**:
- `Dict`: Tracking results with frame-by-frame data

**Example**:
```python
tracker = DeepSORTVideoTracker('yolov8n.pt')
results = tracker.process_video('video.mp4', 'output.mp4')
```

---

### `MOTEvaluator.evaluate(pred_tracks, gt_tracks)`

**Location**: `src/evaluation/mot_metrics.py`

**Purpose**: Calculate MOT metrics

**Parameters**:
- `pred_tracks` (List): Predicted tracks
- `gt_tracks` (List): Ground truth tracks

**Returns**:
- `Dict`: Metrics including MOTA, MOTP, IDF1, etc.

**Metrics**:
- MOTA: Multiple Object Tracking Accuracy
- MOTP: Multiple Object Tracking Precision
- IDF1: ID F1 Score
- Precision, Recall
- ID Switches, Fragments

---

## API Functions

### `POST /upload`

**Location**: `src/api/production_api.py`

**Purpose**: Upload video for processing

**Parameters**:
- `file` (File): Video file

**Returns**:
- `Dict`: Upload confirmation with file_id

---

### `POST /track`

**Location**: `src/api/production_api.py`

**Purpose**: Submit tracking job

**Parameters**:
- `file_id` (str): Uploaded file ID
- `config` (Dict, optional): Tracking configuration

**Returns**:
- `Dict`: Job submission confirmation with job_id

---

### `GET /status/{job_id}`

**Location**: `src/api/production_api.py`

**Purpose**: Check job status

**Parameters**:
- `job_id` (str): Job identifier

**Returns**:
- `Dict`: Job status (pending/processing/completed/failed)

---

### `GET /result/{job_id}`

**Location**: `src/api/production_api.py`

**Purpose**: Get tracking results

**Parameters**:
- `job_id` (str): Job identifier

**Returns**:
- `Dict`: Tracking results (video URL, JSON data, metrics)

---

## Training Functions

### `DetectionTrainer.train()`

**Location**: `src/training/train_detector.py`

**Purpose**: Train YOLOv8 on MOT20 dataset

**Parameters**: Configured via YAML config

**Process**:
1. Load MOT20Dataset
2. Initialize YOLOv8 model
3. Train with validation
4. Save checkpoints
5. Log to Weights & Biases

---

### `ReIDTrainer.train()`

**Location**: `src/training/train_reid.py`

**Purpose**: Train ReID model

**Parameters**: Configured via YAML config

**Process**:
1. Load MOT20ReIDDataset
2. Initialize ReIDModel
3. Train with triplet loss
4. Validate with Rank-1 accuracy
5. Save checkpoints

---

### `TransformerTrackerTrainer.train()`

**Location**: `src/training/train_transformer_tracker.py`

**Purpose**: Train Transformer tracker

**Parameters**: Configured via YAML config

**Process**:
1. Load TrackingDataset
2. Initialize TransformerTracker
3. Train with combined loss
4. Validate on MOT metrics
5. Save checkpoints

---

## Function Complexity

| Function | Time Complexity | Space Complexity |
|----------|----------------|------------------|
| `DeepSORT.update()` | O(n*m*k) | O(n*k) |
| `KalmanFilter.predict()` | O(1) | O(1) |
| `KalmanFilter.update()` | O(1) | O(1) |
| `_cascade_match()` | O(n*m + n*m) | O(n*m) |
| `_compute_cost_matrix()` | O(n*m*k) | O(n*m) |
| `_extract_features()` | O(m*H*W) | O(m*d) |
| `_iou_match()` | O(n*m + min(n,m)³) | O(n*m) |

Where:
- `n`: Number of tracks
- `m`: Number of detections
- `k`: Feature gallery size
- `H, W`: Image dimensions
- `d`: Feature dimension

---

*Document Version: 1.0*  
*Last Updated: 2025-11-22*

