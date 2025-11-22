"""Model optimization for inference speed"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict

try:
    import torch.quantization
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("Warning: PyTorch quantization not available")

try:
    from torch.onnx import export as onnx_export
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX export not available")

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("Warning: ONNX Runtime not available")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT not available")


class ModelOptimizer:
    """
    Optimize models for inference speed
    - Quantization
    - ONNX export
    - TensorRT optimization
    """
    def __init__(self, model, input_shape: tuple):
        self.model = model
        self.input_shape = input_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def quantize_dynamic(self, output_path: str):
        """Apply dynamic quantization (CPU optimization)"""
        if not QUANTIZATION_AVAILABLE:
            print("Error: PyTorch quantization not available")
            return None
        
        print("Applying dynamic quantization...")
        
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            torch.save(quantized_model.state_dict(), output_path)
            print(f"Quantized model saved to {output_path}")
            
            return quantized_model
        except Exception as e:
            print(f"Quantization failed: {e}")
            return None
    
    def quantize_static(self, calibration_loader, output_path: str):
        """Apply static quantization (better accuracy)"""
        if not QUANTIZATION_AVAILABLE:
            print("Error: PyTorch quantization not available")
            return None
        
        print("Applying static quantization...")
        
        try:
            # Prepare model
            self.model.eval()
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            
            # Calibrate
            print("Calibrating...")
            with torch.no_grad():
                for batch in calibration_loader:
                    self.model(batch)
            
            # Convert
            torch.quantization.convert(self.model, inplace=True)
            
            torch.save(self.model.state_dict(), output_path)
            print(f"Quantized model saved to {output_path}")
            
            return self.model
        except Exception as e:
            print(f"Static quantization failed: {e}")
            return None
    
    def export_onnx(self, output_path: str, opset_version: int = 13):
        """Export model to ONNX format"""
        if not ONNX_AVAILABLE:
            print("Error: ONNX export not available")
            return
        
        print(f"Exporting to ONNX (opset {opset_version})...")
        
        try:
            dummy_input = torch.randn(self.input_shape)
            
            # Move to same device as model
            if next(self.model.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            
            self.model.eval()
            
            onnx_export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"ONNX model saved to {output_path}")
            
            # Verify
            if ONNXRUNTIME_AVAILABLE:
                self._verify_onnx(output_path, dummy_input)
        except Exception as e:
            print(f"ONNX export failed: {e}")
    
    def _verify_onnx(self, onnx_path: str, test_input: torch.Tensor):
        """Verify ONNX model produces same output"""
        if not ONNXRUNTIME_AVAILABLE:
            return
        
        print("Verifying ONNX model...")
        
        try:
            # PyTorch output
            self.model.eval()
            with torch.no_grad():
                torch_output = self.model(test_input)
                if isinstance(torch_output, tuple):
                    torch_output = torch_output[0]
                torch_output_np = torch_output.cpu().numpy()
            
            # ONNX Runtime output
            ort_session = ort.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
            ort_output = ort_session.run(None, ort_inputs)[0]
            
            # Compare
            diff = np.abs(torch_output_np - ort_output).max()
            print(f"Max difference: {diff}")
            
            if diff < 1e-3:
                print("✓ ONNX model verified successfully!")
            else:
                print("✗ Warning: Large difference detected")
        except Exception as e:
            print(f"ONNX verification failed: {e}")
    
    def convert_tensorrt(self, onnx_path: str, output_path: str, precision: str = 'fp16'):
        """
        Convert ONNX to TensorRT for maximum speed
        Requires TensorRT installation
        """
        if not TENSORRT_AVAILABLE:
            print("Error: TensorRT not available")
            return None
        
        print(f"Converting to TensorRT ({precision})...")
        
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            with trt.Builder(TRT_LOGGER) as builder:
                network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                
                with builder.create_network(network_flags) as network:
                    with trt.OnnxParser(network, TRT_LOGGER) as parser:
                        # Parse ONNX
                        with open(onnx_path, 'rb') as model:
                            if not parser.parse(model.read()):
                                print("ERROR: Failed to parse ONNX")
                                for error in range(parser.num_errors):
                                    print(parser.get_error(error))
                                return None
                        
                        # Configure builder
                        config = builder.create_builder_config()
                        config.max_workspace_size = 1 << 30  # 1GB
                        
                        # Set precision
                        if precision == 'fp16' and builder.platform_has_fast_fp16:
                            config.set_flag(trt.BuilderFlag.FP16)
                        elif precision == 'int8' and builder.platform_has_fast_int8:
                            config.set_flag(trt.BuilderFlag.INT8)
                        
                        # Build engine
                        print("Building TensorRT engine (this may take a while)...")
                        engine = builder.build_engine(network, config)
                        
                        if engine is None:
                            print("ERROR: Failed to build TensorRT engine")
                            return None
                        
                        # Serialize
                        with open(output_path, 'wb') as f:
                            f.write(engine.serialize())
                        
                        print(f"TensorRT engine saved to {output_path}")
                        
                        return engine
        except Exception as e:
            print(f"TensorRT conversion failed: {e}")
            return None
    
    def benchmark_speed(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference speed"""
        print(f"Benchmarking speed ({num_iterations} iterations)...")
        
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU (slower)")
            return self._benchmark_cpu(num_iterations)
        
        self.model.eval()
        dummy_input = torch.randn(self.input_shape).cuda()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
                if isinstance(_, tuple):
                    _ = _[0]
        
        # Benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
                if isinstance(_, tuple):
                    _ = _[0]
        end.record()
        
        torch.cuda.synchronize()
        
        elapsed_time = start.elapsed_time(end) / 1000  # Convert to seconds
        fps = num_iterations / elapsed_time
        avg_latency = elapsed_time / num_iterations * 1000  # ms
        
        print(f"Average FPS: {fps:.2f}")
        print(f"Average Latency: {avg_latency:.2f} ms")
        
        return {'fps': fps, 'latency': avg_latency}
    
    def _benchmark_cpu(self, num_iterations: int) -> Dict[str, float]:
        """Benchmark on CPU"""
        import time
        
        self.model.eval()
        dummy_input = torch.randn(self.input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
        elapsed_time = time.time() - start
        
        fps = num_iterations / elapsed_time
        avg_latency = elapsed_time / num_iterations * 1000  # ms
        
        print(f"Average FPS: {fps:.2f}")
        print(f"Average Latency: {avg_latency:.2f} ms")
        
        return {'fps': fps, 'latency': avg_latency}


# Usage example
if __name__ == '__main__':
    try:
        from ultralytics import YOLO
        
        # Load model
        model_path = 'models/checkpoints/yolov8n.pt'
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            print("Please train or download a model first")
            exit(1)
        
        model = YOLO(model_path)
        
        # Initialize optimizer
        optimizer = ModelOptimizer(
            model=model.model,
            input_shape=(1, 3, 640, 640)
        )
        
        # Original speed
        print("\n=== Original Model ===")
        original_speed = optimizer.benchmark_speed()
        
        # Export to ONNX
        onnx_path = 'models/checkpoints/yolov8n.onnx'
        optimizer.export_onnx(onnx_path)
        
        # Quantize
        if QUANTIZATION_AVAILABLE:
            quantized_path = 'models/checkpoints/yolov8n_quantized.pth'
            quantized_model = optimizer.quantize_dynamic(quantized_path)
            
            if quantized_model:
                # Benchmark quantized
                print("\n=== Quantized Model ===")
                optimizer.model = quantized_model
                quantized_speed = optimizer.benchmark_speed()
                
                # Summary
                print("\n=== Speed Comparison ===")
                print(f"Original: {original_speed['fps']:.2f} FPS")
                print(f"Quantized: {quantized_speed['fps']:.2f} FPS")
                if original_speed['fps'] > 0:
                    print(f"Speedup: {quantized_speed['fps']/original_speed['fps']:.2f}x")
        
        # Convert to TensorRT (if available)
        if TENSORRT_AVAILABLE and Path(onnx_path).exists():
            try:
                optimizer.convert_tensorrt(
                    onnx_path,
                    'models/checkpoints/yolov8n.trt',
                    precision='fp16'
                )
            except Exception as e:
                print(f"TensorRT conversion skipped: {e}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a trained model available")


