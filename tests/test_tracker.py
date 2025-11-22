"""
Unit tests for tracking system
Implements Quick Wins from docs/5.md
"""
import unittest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tracking.deepsort import DeepSORT, KalmanFilter, Track
from src.models.tracking.simple_tracker import SimpleTracker
from src.models.detection.yolo_detector import YOLODetector


class TestKalmanFilter(unittest.TestCase):
    """Test Kalman Filter implementation"""
    
    def setUp(self):
        self.kf = KalmanFilter()
    
    def test_initiate(self):
        """Test track initiation"""
        measurement = np.array([100, 100, 0.5, 50])
        mean, covariance = self.kf.initiate(measurement)
        
        self.assertEqual(mean.shape, (8,))
        self.assertEqual(covariance.shape, (8, 8))
    
    def test_predict(self):
        """Test prediction step"""
        measurement = np.array([100, 100, 0.5, 50])
        mean, covariance = self.kf.initiate(measurement)
        
        mean_pred, covariance_pred = self.kf.predict(mean, covariance)
        
        self.assertEqual(mean_pred.shape, (8,))
        self.assertEqual(covariance_pred.shape, (8, 8))
    
    def test_update(self):
        """Test update step"""
        measurement = np.array([100, 100, 0.5, 50])
        mean, covariance = self.kf.initiate(measurement)
        mean, covariance = self.kf.predict(mean, covariance)
        
        new_measurement = np.array([105, 105, 0.5, 50])
        mean_updated, covariance_updated = self.kf.update(mean, covariance, new_measurement)
        
        self.assertEqual(mean_updated.shape, (8,))
        self.assertEqual(covariance_updated.shape, (8, 8))


class TestTrack(unittest.TestCase):
    """Test Track class"""
    
    def setUp(self):
        self.track = Track(
            mean=np.array([100, 100, 0.5, 50, 0, 0, 0, 0]),
            covariance=np.eye(8),
            track_id=1
        )
    
    def test_track_creation(self):
        """Test track initialization"""
        self.assertEqual(self.track.track_id, 1)
        self.assertEqual(self.track.age, 1)
        self.assertEqual(self.track.time_since_update, 0)
    
    def test_to_tlwh(self):
        """Test bounding box conversion"""
        bbox = self.track.to_tlwh()
        self.assertEqual(len(bbox), 4)
    
    def test_to_tlbr(self):
        """Test bounding box conversion"""
        bbox = self.track.to_tlbr()
        self.assertEqual(len(bbox), 4)


class TestSimpleTracker(unittest.TestCase):
    """Test Simple Tracker"""
    
    def setUp(self):
        self.tracker = SimpleTracker()
    
    def test_update(self):
        """Test tracker update"""
        detections = np.array([
            [100, 100, 200, 200, 0.9],
            [300, 300, 400, 400, 0.8]
        ])
        
        tracks = self.tracker.update(detections)
        
        self.assertGreater(len(tracks), 0)
        self.assertEqual(len(tracks[0]), 5)  # x1, y1, x2, y2, track_id


class TestDeepSORT(unittest.TestCase):
    """Test DeepSORT tracker"""
    
    def setUp(self):
        # Create dummy ReID model
        self.reid_model = None  # Can be None for basic tests
        self.tracker = DeepSORT(
            reid_model=self.reid_model,
            max_dist=0.2,
            max_iou_distance=0.7,
            max_age=30,
            n_init=3
        )
    
    def test_update(self):
        """Test DeepSORT update"""
        detections = np.array([
            [100, 100, 200, 200, 0.9],
            [300, 300, 400, 400, 0.8]
        ])
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        tracks = self.tracker.update(detections, frame)
        
        # Should return tracks
        self.assertIsInstance(tracks, list)


class TestYOLODetector(unittest.TestCase):
    """Test YOLO Detector"""
    
    def setUp(self):
        try:
            self.detector = YOLODetector(model_path='yolov8n.pt')
        except Exception as e:
            self.skipTest(f"YOLO model not available: {e}")
    
    def test_detect(self):
        """Test detection"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = self.detector.detect(frame)
        
        self.assertIsInstance(detections, np.ndarray)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestKalmanFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestTrack))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestDeepSORT))
    suite.addTests(loader.loadTestsFromTestCase(TestYOLODetector))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

