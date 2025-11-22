# 快速开始指南 - 人员跟踪系统

几分钟内开始使用人员跟踪系统。

## 前置要求

- Python 3.8+
- pip
- （可选）支持 CUDA 的 GPU 以加速处理

## 安装

### 1. 克隆和设置

```bash
# 进入项目目录
cd people-tracking-system

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型

```bash
# 下载 YOLOv8 检测模型（如果不存在会自动下载）
python scripts/download_models.py
```

模型将保存到 `models/checkpoints/yolov8n.pt`。

## 快速测试

### 使用摄像头测试

```bash
python -m src.inference.main --input 0 --display
```

按 `q` 退出。

### 使用视频文件测试

```bash
python -m src.inference.main --input path/to/video.mp4 --output output.mp4
```

## 常用用法

### 1. 对视频进行跟踪

```bash
python -m src.inference.main \
    --input data/raw/video.mp4 \
    --output data/processed/output.mp4 \
    --config configs/tracking_config.yaml
```

### 2. 启动 API 服务器

```bash
python -m src.api.main
```

然后访问 `http://localhost:8000/docs` 查看 API 文档。

### 3. 启动 Web 界面

```bash
python -m src.ui.main
```

然后在浏览器中访问 `http://localhost:8501`。

## 配置

编辑 `configs/tracking_config.yaml` 以调整：

- 检测阈值 (`detection.conf_threshold`)
- 跟踪参数 (`tracking.max_dist`, `tracking.max_age`)
- 模型路径

## 示例：处理 MOT20 序列

```bash
# 处理 MOT20 序列
python -m src.inference.main \
    --input data/raw/MOT20/MOT20/train/MOT20-01/img1 \
    --output data/processed/MOT20-01_tracked.mp4 \
    --config configs/tracking_config.yaml
```

## 故障排除

### 导入错误

如果看到 `ModuleNotFoundError`，请确保：
1. 虚拟环境已激活
2. 依赖已安装：`pip install -r requirements.txt`

### GPU 未检测到

系统可在 CPU 上运行，但速度较慢。要使用 GPU：
1. 安装带 CUDA 的 PyTorch：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
2. 验证：`python -c "import torch; print(torch.cuda.is_available())"`

### 模型下载问题

模型在首次使用时自动下载。如有问题：
- 检查网络连接
- 手动从 Ultralytics 下载：https://github.com/ultralytics/ultralytics

## 下一步

- **训练**：查看 `src/training/` 了解模型训练脚本
- **评估**：运行 `python scripts/evaluate_benchmark.py` 获取指标
- **部署**：查看 `docker-compose.yml` 了解 Docker 设置
- **文档**：查看 `docs-zh/` 获取详细中文文档

## 需要帮助？

- 查看 `README.md` 获取完整文档
- 查看 `configs/tracking_config.yaml` 了解配置选项
- 查看 `src/inference/main.py` 了解 CLI 选项

---

**准备开始跟踪！** 🚀

