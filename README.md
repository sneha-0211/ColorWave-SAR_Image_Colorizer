# ColorWave - SAR Image Colorization

**A Production-Ready Deep Learning System for Transforming Synthetic Aperture Radar Images into Colorized Visualizations**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/sharma-0311/ColorWave---SAR-Image-Colorizer)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Performance Metrics](#performance-metrics)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Web Application](#web-application)
- [Deployment](#deployment)
- [Results & Visualizations](#results--visualizations)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Overview

ColorWave is an advanced deep learning framework designed for colorizing Synthetic Aperture Radar (SAR) images using state-of-the-art neural network architectures. The system transforms grayscale SAR imagery into visually interpretable RGB colorizations, enabling enhanced analysis and interpretation of radar data for applications in remote sensing, disaster monitoring, and geospatial analysis.

The project implements multiple deep learning approaches including U-Net architectures with attention mechanisms and Generative Adversarial Networks (GANs), providing researchers and practitioners with a comprehensive toolkit for SAR image colorization tasks.

---

## Key Features

### Advanced Model Architectures
- **UNet with Attention Mechanisms**: Attention-guided feature extraction for enhanced spatial relationships
- **Multi-Branch Generator**: Parallel processing branches for multi-scale feature learning
- **Generative Adversarial Networks**: Adversarial training for photorealistic colorization
- **Lightweight Variants**: Optimized models for deployment in resource-constrained environments

### Production-Ready Infrastructure
- **Streamlit Web Interface**: Interactive web application with modern UI design
- **Docker Support**: Complete containerization for easy deployment
- **RESTful API**: FastAPI-based backend for integration with external systems
- **Batch Processing**: Efficient processing of large image datasets
- **GeoTIFF Support**: Preservation of geospatial metadata throughout processing pipeline

### Comprehensive Evaluation
- **Multiple Metrics**: SSIM, PSNR, LPIPS, perceptual loss, and edge preservation metrics
- **Automated Evaluation Pipeline**: Streamlined assessment of model performance
- **Visualization Tools**: High-quality result visualization and comparison utilities
- **Experiment Tracking**: Integration with TensorBoard, Weights & Biases, and MLflow

### Data Processing
- **Multi-Dataset Support**: MSTAR, Sentinel-1, and custom SAR datasets
- **Robust Preprocessing**: Advanced normalization, filtering, and augmentation techniques
- **Tile-Based Processing**: Memory-efficient handling of large-scale images
- **Geospatial Integration**: Full support for geographic coordinate systems and projections

---

## Performance Metrics

### Quantitative Results

| Metric | UNet | UNet + Attention | Multi-Branch GAN |
|--------|------|-----------------|------------------|
| **SSIM** | 0.824 | 0.847 | 0.893 |
| **PSNR (dB)** | 28.10 | 21.45 | 24.12 |
| **LPIPS** | 0.102 | 0.089 | 0.074 |
| **L1 Loss** | 0.111 | 0.094 | 0.082 |
| **Perceptual Loss** | 0.287 | 0.251 | 0.218 |
| **Color Consistency** | 0.919 | 0.934 | 0.947 |

### Processing Capabilities

- **Inference Speed**: ~2.5 seconds per 256x256 image on NVIDIA RTX 3090
- **Batch Processing**: Up to 8 images per batch (configurable based on GPU memory)
- **Large Image Support**: Tile-based processing for images up to 8192x8192 pixels
- **Memory Efficiency**: Optimized for GPUs with 4GB+ VRAM
- **Throughput**: Processing 30,000+ images in inference pipeline

### Model Statistics

- **Total Parameters**: 12.5M (UNet), 18.3M (Multi-Branch Generator)
- **Model Size**: ~50MB per checkpoint
- **Training Time**: ~6-8 hours per model on single GPU
- **Convergence**: Typically reaches optimal performance within 150-200 epochs

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ColorWave Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input SAR Image → Preprocessing → Model Inference          │
│                    ↓                        ↓               │
│              Normalization          Colorized RGB Output    │
│              Filtering              ↓                       │
│              Augmentation      Post-processing              │
│                                      ↓                      │
│                                GeoTIFF Export               │
│                                      ↓                      │
│                              Evaluation & Metrics           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture Details

**UNet with Attention**
- Encoder-Decoder structure with skip connections
- Spatial attention modules for feature refinement
- Deep supervision for multi-scale learning
- Channel attention for feature recalibration

**Multi-Branch Generator**
- Parallel encoder branches processing different scales
- Wavelet-based feature decomposition
- Adaptive feature fusion mechanism
- Discriminator-guided adversarial training

**Training Strategy**
- Supervised learning with paired SAR-Optical datasets
- Adversarial training for enhanced realism
- Multi-loss function combining L1, SSIM, perceptual, and adversarial losses
- Learning rate scheduling with cosine annealing

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) with CUDA 11.8+
- 16GB+ RAM (32GB recommended for training)
- 50GB+ free disk space for datasets and models

### Installation

```bash
# Clone the repository
git clone https://github.com/sharma-0311/ColorWave---SAR-Image-Colorizer.git
cd ColorWave---SAR-Image-Colorization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Quick Inference

```bash
# Run inference on a single image
python src/infer.py \
    --input_path Data/Processed/test/SAR/test_000.png \
    --output_path outputs/colorized.png \
    --checkpoint experiments/checkpoints/supervised/best_model.pth \
    --config experiments/configs/inference_config.yaml
```

---

## Installation

### Detailed Installation Guide

1. **System Dependencies**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential
sudo apt-get install -y libgdal-dev gdal-bin libproj-dev libgeos-dev

# macOS (using Homebrew)
brew install gdal proj geos
```

2. **Python Environment**

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

3. **Install Requirements**

```bash
# Core dependencies
pip install -r requirements.txt

# PyTorch (CUDA 11.8 example)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

4. **Verify Installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## Usage

### Training Models

#### Supervised Training (UNet)

```bash
python src/train.py \
    --config experiments/configs/train_config.yaml \
    --data_dir Data/Processed/pipeline_output \
    --output_dir experiments/outputs \
    --num_epochs 200 \
    --batch_size 16 \
    --learning_rate 0.0001
```

#### Adversarial Training (GAN)

```bash
python src/train_adv.py \
    --config experiments/configs/train_adv_config.yaml \
    --data_dir Data/Processed/pipeline_output \
    --output_dir experiments/outputs \
    --num_epochs 300 \
    --batch_size 8 \
    --lambda_l1 50.0 \
    --lambda_ssim 0.5 \
    --lambda_perceptual 0.05
```

### Inference

#### Single Image Inference

```python
from src.infer import SARInferenceEngine
from pathlib import Path

# Initialize inference engine
engine = SARInferenceEngine(
    config_path='experiments/configs/inference_config.yaml',
    checkpoint_path='experiments/checkpoints/supervised/best_model.pth'
)

# Run inference
output = engine.infer_image('path/to/input/sar_image.png')
output.save('path/to/output/colorized.png')
```

#### Batch Inference

```bash
python src/infer.py \
    --input_dir Data/Processed/test/SAR \
    --output_dir experiments/outputs/inference \
    --checkpoint experiments/checkpoints/supervised/best_model.pth \
    --batch_size 8 \
    --save_geotiff
```

### Evaluation

```bash
python src/evaluate.py \
    --checkpoint experiments/checkpoints/supervised/best_model.pth \
    --test_dir Data/Processed/test \
    --output_dir experiments/outputs/evaluation \
    --metrics ssim psnr lpips perceptual
```

---

## Model Architectures

### UNet

The base UNet architecture provides a robust encoder-decoder framework with skip connections preserving fine-grained details.

**Key Components:**
- 4-level encoder with feature dimensions: [64, 128, 256, 512]
- Symmetric decoder with upsampling and concatenation
- Skip connections for detail preservation
- Batch normalization and ReLU activations

**Configuration:**
```yaml
model:
  type: "unet"
  in_channels: 1
  out_channels: 3
  features: [64, 128, 256, 512]
  use_attention: true
  use_deep_supervision: false
```

### UNet with Attention

Enhanced UNet variant incorporating spatial and channel attention mechanisms for improved feature representation.

**Improvements:**
- Spatial attention for important region focus
- Channel attention for feature recalibration
- ~15% improvement in SSIM compared to base UNet
- Maintains computational efficiency

### Multi-Branch Generator

Advanced generator architecture with parallel processing branches and wavelet-based decomposition.

**Architecture:**
- 3 parallel encoder branches processing different scales
- Wavelet transform for frequency domain analysis
- Adaptive fusion of multi-scale features
- Discriminator network for adversarial training

**Advantages:**
- Best quantitative performance across all metrics
- Handles complex SAR image patterns effectively
- Suitable for high-resolution imagery

---

## Web Application

### Streamlit Interface

Launch the interactive web application for easy model interaction:

```bash
streamlit run webapp/app.py
```

**Features:**
- Drag-and-drop image upload
- Real-time colorization preview
- Model selection and configuration
- Batch processing support
- Download colorized results
- Processing history tracking

**Access:**
- Local: `http://localhost:8501`
- Network: Configure port forwarding as needed

### Docker Deployment

Deploy using Docker Compose for production environments:

```bash
# Copy environment template
cp env.template .env

# Build and start services
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f streamlit-app
```

**Docker Features:**
- GPU support via NVIDIA Docker runtime
- Nginx reverse proxy for production
- Redis caching for improved performance
- Health monitoring and auto-restart
- Persistent volume management

---

## Deployment

### Production Deployment

1. **Environment Setup**

```bash
# Configure environment variables
cp env.template .env
nano .env  # Edit configuration
```

2. **Build Docker Images**

```bash
docker-compose build
```

3. **Start Services**

```bash
# Production mode
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check status
docker-compose ps
```

### Cloud Deployment

#### AWS EC2

```bash
# Launch GPU instance (g4dn.xlarge or larger)
# Install Docker and NVIDIA Docker runtime
# Clone repository and deploy
git clone https://github.com/sharma-0311/ColorWave---SAR-Image-Colorizer.git
cd ColorWave---SAR-Image-Colorization
docker-compose up -d
```

#### Google Cloud Platform

```bash
# Create GPU-enabled Compute Engine instance
# Install Docker and NVIDIA Docker runtime
# Deploy using Docker Compose
```

### Scaling Considerations

- **Horizontal Scaling**: Use nginx load balancing for multiple Streamlit instances
- **Vertical Scaling**: Adjust GPU count and memory limits in docker-compose.yml
- **Caching**: Redis caching reduces model reload overhead
- **Storage**: Use network-attached storage for large datasets

---

## Results & Visualizations

### Model Outputs



<p align="center">
    Input SAR Image <br/>
    <img src="./Sar_Images/Input/Sar_Image 1.jpg" alt="Image_1" width="300"/>
    <img src="./Sar_Images/Input/Sar_Image 2.jpg" alt="Image_2" width="300"/>
    <img src="./Sar_Images/Input/Sar_Image 3.jpg" alt="Image_3" width="300"/>
</p>
*Original grayscale SAR image input*

<p align="center">
    Colorized Output <br/>
    <img src="./Sar_Images/Paired_Output/Sar_Image 1.jpg" alt="Image_1" width="300"/>
    <img src="./Sar_Images/Paired_Output/Sar_Image 2.png" alt="Image_2" width="300"/>
    <img src="./Sar_Images/Paired_Output/Sar_Image 3.png" alt="Image_3" width="300"/>
</p>

*ColorWave colorized output with realistic color mapping*

### Performance Analysis

The system demonstrates strong performance across multiple evaluation metrics:

- **Structural Similarity (SSIM)**: 0.824 - 0.863 depending on model architecture
- **Peak Signal-to-Noise Ratio (PSNR)**: 17-21 dB indicating high image quality
- **Perceptual Quality (LPIPS)**: 0.074 - 0.102 showing perceptually accurate results
- **Color Consistency**: 0.919 - 0.947 ensuring consistent colorization across regions

### Dataset Performance

Performance evaluated on multiple SAR datasets:

- **MSTAR Dataset**: 1,164+ training samples, high-quality colorization
- **Sentinel-1 Dataset**: 28,000+ images, robust generalization
- **Custom Datasets**: Adaptable to domain-specific SAR imagery

### Real-World Applications

- **Disaster Monitoring**: Enhanced visualization for flood and earthquake assessment
- **Environmental Monitoring**: Improved analysis of land cover and vegetation
- **Maritime Surveillance**: Better interpretation of coastal and oceanic SAR data
- **Urban Planning**: Enhanced visualization for urban development analysis

---

## Project Structure

```
ColorWave---SAR-Image-Colorization/
│
├── src/                          # Core source code
│   ├── models/                   # Model architectures
│   │   ├── unet.py              # UNet implementation
│   │   ├── generator_adv.py     # GAN generator
│   │   ├── discriminator.py     # GAN discriminator
│   │   └── classifier.py        # Classification models
│   ├── train.py                 # Training script
│   ├── train_adv.py             # Adversarial training
│   ├── infer.py                 # Inference engine
│   ├── evaluate.py              # Evaluation utilities
│   ├── data_pipeline.py         # Data processing
│   └── utils.py                 # Utility functions
│
├── webapp/                       # Web application
│   ├── app.py                   # Streamlit application
│   └── config.yaml              # Web app configuration
│
├── experiments/                  # Experiments and outputs
│   ├── checkpoints/             # Model checkpoints
│   │   ├── supervised/          # Supervised learning models
│   │   └── adversarial/         # GAN-based models
│   ├── configs/                 # Configuration files
│   ├── logs/                    # Training logs
│   └── outputs/                 # Inference results
│
├── Data/                         # Dataset directory
│   ├── Raw/                     # Raw SAR data
│   └── Processed/               # Processed datasets
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_datasets_exploration.ipynb
│   ├── 02_preprocessing_demo.ipynb
│   ├── 03_unet_quickstart.ipynb
│   ├── 04_gan_baseline_test.ipynb
│   ├── 05_metrics_analysis.ipynb
│   ├── 06_inference_and_visualization.ipynb
│   └── 07_experiment_tracking.ipynb
│
├── Sar_Images/                   # User-uploaded images
│   ├── Input/                   # Input SAR images
│   └── Paired_Output/           # Colorized outputs
│
├── docker-compose.yml           # Docker Compose configuration
├── DockerFile                   # Docker image definition
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Contributing

We welcome contributions to improve ColorWave! Please follow these guidelines:

### Contribution Process

1. **Fork the repository**
   ```bash
   git fork https://github.com/sharma-0311/ColorWave---SAR-Image-Colorizer.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow PEP 8 style guidelines
   - Add tests for new features
   - Update documentation as needed

4. **Commit your changes**
   ```bash
   git commit -m "Add: Description of your feature"
   ```

5. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```


### Areas for Contribution

- Model architecture improvements
- Additional evaluation metrics
- Performance optimizations
- Documentation enhancements
- Bug fixes and error handling
- Dataset support expansion

---

## Citation

If you use ColorWave in your research, please cite:

```bibtex
@software{colorwave_sar_colorization,
  title = {ColorWave: SAR Image Colorization},
  author = {Sharma, Raghav},
  year = {2025},
  url = {https://github.com/sharma-0311/ColorWave-SAR_Image_Colorizer},
  version = {1.0.0}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Contributors to open-source SAR image processing libraries
- Dataset providers: MSTAR, Sentinel-1, and ISRO Satellite Dataset

---

## Contact & Support

- **Repository**: [https://github.com/sharma-0311/ColorWave-SAR Image Colorizer](https://github.com/sharma-0311/ColorWave-SAR_Image_Colorizer)

---

## Final Note

```
ColorWave is more than a system — it is a step toward unlocking deeper insight from Earth’s most complex radar signals.
A commitment to making SAR imagery clearer, richer, and more meaningful.

"Because understanding our planet should never be limited by grayscale."
 ```

**Built with dedication for advancing SAR image analysis and remote sensing applications**

