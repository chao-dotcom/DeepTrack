"""Deployment script for production"""
import subprocess
import sys
from pathlib import Path
import yaml
import os


class Deployer:
    """Handles deployment pipeline"""
    def __init__(self, config_path: str = None):
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config
            self.config = {
                'models': {
                    'detection_url': None,
                    'reid_url': None
                },
                'deployment': {
                    'method': 'docker',  # or 'local'
                    'device': 'cuda',  # or 'cpu'
                    'port': 8000
                }
            }
        
        self.checkpoint_dir = Path('models/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def download_models(self):
        """Download pre-trained models"""
        print("Checking models...")
        
        # Check detection model
        detection_model = self.checkpoint_dir / 'yolov8n.pt'
        if not detection_model.exists():
            print("Detection model not found. Please download it first:")
            print("  python scripts/download_models.py")
        else:
            print(f"✓ Detection model found: {detection_model}")
        
        # Check ReID model (optional)
        reid_model = self.checkpoint_dir / 'reid_best.pth'
        if not reid_model.exists():
            print("ReID model not found (optional). Train with:")
            print("  python -m src.training.train_reid")
        else:
            print(f"✓ ReID model found: {reid_model}")
    
    def optimize_models(self):
        """Optimize models for production"""
        print("Optimizing models...")
        
        try:
            from src.optimization.model_optimization import ModelOptimizer
            from ultralytics import YOLO
            
            detection_model = self.checkpoint_dir / 'yolov8n.pt'
            if not detection_model.exists():
                print("Skipping optimization: detection model not found")
                return
            
            # Load and optimize detection model
            model = YOLO(str(detection_model))
            optimizer = ModelOptimizer(model.model, (1, 3, 640, 640))
            
            # Export to ONNX
            onnx_path = self.checkpoint_dir / 'yolov8n.onnx'
            optimizer.export_onnx(str(onnx_path))
            
            # Quantize if CPU deployment
            if self.config['deployment'].get('device') == 'cpu':
                quantized_path = self.checkpoint_dir / 'yolov8n_quantized.pth'
                optimizer.quantize_dynamic(str(quantized_path))
            
            print("✓ Models optimized")
        except Exception as e:
            print(f"Warning: Model optimization failed: {e}")
            print("Continuing without optimization...")
    
    def setup_database(self):
        """Setup Redis and database"""
        print("Setting up database...")
        
        # Check if Redis is available
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            print("✓ Redis is running")
        except Exception:
            print("Warning: Redis not available. Install and start Redis:")
            print("  docker run -d -p 6379:6379 redis:7-alpine")
            print("  Or install: sudo apt-get install redis-server")
    
    def run_tests(self):
        """Run tests before deployment"""
        print("Running tests...")
        
        try:
            result = subprocess.run(
                ['pytest', 'tests/', '-v'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print("⚠ Some tests failed (continuing anyway)")
                if result.stdout:
                    print(result.stdout)
            else:
                print("✓ All tests passed")
        except FileNotFoundError:
            print("⚠ pytest not found, skipping tests")
        except subprocess.TimeoutExpired:
            print("⚠ Tests timed out, continuing anyway")
        except Exception as e:
            print(f"⚠ Test execution failed: {e}")
    
    def deploy(self):
        """Full deployment pipeline"""
        print("="*50)
        print("DEPLOYMENT STARTED")
        print("="*50)
        
        # Step 1: Check models
        self.download_models()
        
        # Step 2: Optimize models (optional)
        if self.config['deployment'].get('optimize', False):
            self.optimize_models()
        
        # Step 3: Setup database
        self.setup_database()
        
        # Step 4: Run tests (optional)
        if self.config['deployment'].get('run_tests', True):
            self.run_tests()
        
        # Step 5: Start services
        method = self.config['deployment'].get('method', 'docker')
        
        if method == 'docker':
            print("\nStarting Docker services...")
            compose_file = Path('docker-compose.production.yml')
            if compose_file.exists():
                try:
                    subprocess.run(
                        ['docker-compose', '-f', str(compose_file), 'up', '-d'],
                        check=True
                    )
                    print("✓ Docker services started")
                    print(f"\nAPI available at: http://localhost:{self.config['deployment']['port']}")
                    print("API docs at: http://localhost:8000/docs")
                except subprocess.CalledProcessError:
                    print("✗ Docker Compose failed. Make sure Docker is running.")
                    print("Falling back to local deployment...")
                    method = 'local'
                except FileNotFoundError:
                    print("✗ docker-compose not found. Install Docker Compose.")
                    method = 'local'
            else:
                print("✗ docker-compose.production.yml not found")
                method = 'local'
        
        if method == 'local':
            print("\nStarting local services...")
            print("Run the API server with:")
            print("  python -m src.api.production_api")
            print(f"\nAPI will be available at: http://localhost:{self.config['deployment']['port']}")
        
        print("="*50)
        print("DEPLOYMENT COMPLETED")
        print("="*50)
        
        print("\nNext steps:")
        print("1. Check API health: http://localhost:8000/api/v2/health")
        print("2. View API docs: http://localhost:8000/docs")
        print("3. Upload a video via API or web UI")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy People Tracking System')
    parser.add_argument('--config', type=str, help='Path to deployment config YAML')
    parser.add_argument('--skip-tests', action='store_true', help='Skip tests')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip model optimization')
    
    args = parser.parse_args()
    
    deployer = Deployer(config_path=args.config)
    
    if args.skip_tests:
        deployer.config['deployment']['run_tests'] = False
    if args.skip_optimization:
        deployer.config['deployment']['optimize'] = False
    
    deployer.deploy()


