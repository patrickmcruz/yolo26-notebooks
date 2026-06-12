# 01-setup: Environment Configuration & Hardware Analysis

## Overview

This directory contains notebooks for initial system setup, environment validation, and hardware capability analysis.

**👉 Start here**: Run these notebooks FIRST before any training or testing.

## Notebooks

### 1. `setup-density-hardware-analysis.ipynb`

**Purpose**: Analyze your system hardware and generate optimization recommendations.

**What it does**:
- Detects CPU count and specifications
- Measures available RAM and current usage
- Identifies GPU(s) and CUDA compatibility
- Reports VRAM per GPU
- Recommends optimal batch sizes for your hardware
- Suggests image sizes and inference speeds

**Prerequisites**:
- Python 3.8+
- PyTorch with CUDA support (if GPU available)
- psutil package

**Expected outputs**:
- System specifications summary
- GPU/CUDA information
- Recommended parameters for training
- Batch size recommendations

**Runtime**: ~5-10 minutes

**Key metrics generated**:
- Available VRAM (critical for model selection)
- Optimal batch size for your GPU
- Recommended image input size (960x960 or 1280x1280)
- Estimated inference speed

### 2. `test-setup-01.ipynb`

**Purpose**: Validate complete environment setup end-to-end.

**What it does**:
- Verifies Python version
- Checks installed packages (ultralytics, torch, torchvision, roboflow)
- Tests CUDA availability and GPU detection
- Loads a pre-trained YOLO model
- Runs inference on sample image
- Disables telemetry

**Prerequisites**:
- Python 3.8+
- All packages in config.yaml installed
- Sample image (downloads automatically if needed)

**Expected outputs**:
- Setup validation report
- Model loading confirmation
- Inference test result
- Hardware summary

**Runtime**: ~2-3 minutes (longer if downloading model weights)

**Success indicators**:
- ✅ All packages imported successfully
- ✅ Model loaded without errors
- ✅ Inference completes and produces predictions
- ✅ Setup complete message displayed

## Configuration

**File**: `config.yaml`

```yaml
category: setup
description: Environment setup and hardware analysis

dependencies:
  - ultralytics>=8.4.0
  - torch>=2.0.0
  - torchvision>=0.15.0
  - supervision
  - roboflow
  - numpy
  - pandas
  - opencv-python
  - psutil
  - pyyaml

hardware_requirements:
  min_vram_gb: 4
  recommended_vram_gb: 8
  gpu_required: false
  cpu_cores_min: 2

pytorch_installation: |
  # For CUDA 11.8:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  
  # For CPU only:
  pip install torch torchvision torchaudio

estimated_runtime_minutes: 15

expected_outputs:
  - System specifications printed to console
  - GPU/CUDA information
  - Recommended parameters for training
  - Model weights file (yolo26m.pt, ~50MB)
```

## Installation Instructions

### Step 1: Install Python Packages

```bash
# Core dependencies
pip install -q ultralytics>=8.4.0 supervision roboflow
pip install numpy pandas opencv-python psutil pyyaml

# PyTorch with CUDA support (recommended for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only PyTorch (for testing without GPU)
pip install torch torchvision torchaudio
```

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
```

### Step 3: Run Setup Notebooks

```bash
# First: Analyze hardware
jupyter notebook setup-density-hardware-analysis.ipynb

# Second: Validate setup
jupyter notebook test-setup-01.ipynb
```

## Troubleshooting

### Problem: "CUDA not available"
**Solution**: 
1. Check if you have an NVIDIA GPU: `nvidia-smi`
2. Install CUDA toolkit and cuDNN from NVIDIA website
3. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
   ```

### Problem: "No module named 'torch'"
**Solution**: Install PyTorch:
```bash
pip install torch torchvision torchaudio
```

### Problem: "Out of memory" errors
**Solution**:
1. Run `setup-density-hardware-analysis.ipynb` to get recommended batch size
2. Close other GPU-consuming applications
3. Use a smaller model (yolo26n instead of yolo26x)

### Problem: Model weights download timeout
**Solution**:
1. Check internet connection
2. Manually download from: https://github.com/ultralytics/assets/releases
3. Place in current working directory

## Next Steps

After completing setup validation:

1. **For Training**: 
   → Go to `02-training/` directory
   → Follow the training guide in that README

2. **For Testing**:
   → Go to `03-testing/` directory
   → Start with quick validation in `04-validation/`

3. **For Optimization**:
   → Review `hardware_profile.json` for optimal parameters
   → Use these parameters when training

## Environment Variables (Optional)

Set these for custom behavior:

```bash
# Disable Ultralytics telemetry
export YOLOv5_DISABLE_ANALYTICS=1

# Set device (cuda:0, cpu, etc)
export CUDA_VISIBLE_DEVICES=0

# Reduce verbosity
export YOLO_VERBOSE=0
```

## Hardware Profiles

Common configurations and their optimal parameters:

| GPU | VRAM | Batch Size | Image Size | Max FPS |
|-----|------|------------|------------|---------|
| RTX 3090 | 24 GB | 64 | 1280 | 60+ |
| RTX 4090 | 24 GB | 64 | 1280 | 100+ |
| RTX 3070 | 8 GB | 16 | 960 | 30 |
| RTX 3060 | 12 GB | 32 | 1024 | 25 |
| CPU Only | - | 8 | 640 | 1-2 |

**Note**: Run `setup-density-hardware-analysis.ipynb` for your specific configuration!

---

**Status**: ✅ Complete  
**Last Updated**: 2026-05-26  
**Next Directory**: See `02-training/README.md`
