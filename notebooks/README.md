# Jupyter Notebooks - Organized by Function

Welcome to the notebooks collection! All notebooks are organized by their function (setup, training, testing, validation).

## 📁 Directory Structure

```
notebooks/
├── setup/              # Environment setup & hardware analysis
├── training/           # Model training workflows  
├── testing/            # Comprehensive test suites
└── validation/         # Quick sanity checks
```

## 🚀 Quick Start

### 1. Setup Your Environment (First Time - 15 min)

```bash
cd notebooks/setup/
# Read: 01-SETUP-README.md
# Run: setup-density-hardware-analysis.ipynb
# Run: test-setup-01.ipynb
```

### 2. Train a Model (60-90 min)

```bash
cd notebooks/training/
# Read: 02-TRAINING-README.md
# Prepare your dataset
# Run appropriate training notebook
```

### 3. Test & Optimize (30 min)

```bash
cd notebooks/testing/
# Read: 03-TESTING-README.md
# Run test suites in order
```

### 4. Validate (1 min)

```bash
cd notebooks/validation/
# Run: test-quick-validation.ipynb
```

## 📚 Notebooks by Category

### setup/ (2 notebooks)

| Notebook | Purpose | Time |
|----------|---------|------|
| setup-density-hardware-analysis.ipynb | Analyze GPU/CPU and recommend parameters | 10 min |
| test-setup-01.ipynb | Validate environment and dependencies | 5 min |

**Prerequisites**: Python 3.8+  
**Read**: 01-SETUP-README.md

### training/ (4 notebooks + resources)

| Notebook | Purpose | Time |
|----------|---------|------|
| _train-template.ipynb | Template for creating new training workflows | - |
| train-yolo26-object-detection-on-custom-dataset.ipynb | Train object detection models | 60 min |
| train-yolo26-instance-segmentation-on-custom-dataset.ipynb | Train segmentation models | 90 min |
| nvidia-launchables/how-to-finetune-rf-detr-on-segmentation-dataset-a100.ipynb | Advanced DETR fine-tuning (A100 GPU) | 120 min |

**Prerequisites**: Setup validation complete, 8GB+ VRAM  
**Read**: 02-TRAINING-README.md

### testing/ (3 test suites)

| Directory | Purpose | Time |
|-----------|---------|------|
| test-setup-02-train-hparams/ | Hyperparameter optimization | 40 min |
| test-setup-03-run-few/ | Quick inference test (few-shot) | 5 min |
| test-setup-04-run-density/ | High-density scenario testing | 10 min |

**Prerequisites**: Trained model or weights available  
**Read**: 03-TESTING-README.md

### validation/ (1 notebook)

| Notebook | Purpose | Time |
|----------|---------|------|
| test-quick-validation.ipynb | End-to-end pipeline validation | 1 min |

**Prerequisites**: Model available  
**Read**: 04-VALIDATION-README.md  
**Use case**: CI/CD pipelines, pre-deployment checks

## 📖 Documentation

Each directory contains:

- **README.md** - Detailed guide with procedures, prerequisites, troubleshooting
- **config.yaml** - Configuration file with parameters, dependencies, requirements
- **Notebooks** - Cell-level documentation with docstrings and comments

## 🔄 Typical Workflows

### Complete Pipeline (Train → Test → Deploy)
```
setup/ → training/ → testing/ → validation/ → deploy
```

### Quick Validation Only
```
validation/ → decision (deploy/iterate)
```

### Optimization Pipeline
```
setup/ → training/ → testing/ → re-train → validation/
```

### CI/CD Pipeline
```
validation/ → pass/fail → deploy/block
```

## 📋 File Organization

```
notebooks/
├── setup/
│   ├── 01-SETUP-README.md
│   ├── CONFIG-SETUP.yaml
│   ├── setup-density-hardware-analysis.ipynb
│   └── test-setup-01.ipynb
│
├── training/
│   ├── 02-TRAINING-README.md
│   ├── CONFIG-TRAINING.yaml
│   ├── _train-template.ipynb
│   ├── train-yolo26-object-detection-on-custom-dataset.ipynb
│   ├── train-yolo26-instance-segmentation-on-custom-dataset.ipynb
│   └── nvidia-launchables/
│       └── how-to-finetune-rf-detr-on-segmentation-dataset-a100.ipynb
│
├── testing/
│   ├── 03-TESTING-README.md
│   ├── CONFIG-TESTING.yaml
│   ├── test-setup-02-train-hparams/
│   │   └── test-setup-02-train-hparams.ipynb
│   ├── test-setup-03-run-few/
│   │   └── test-setup-03-run-few-yolo26-pose-count.ipynb
│   └── test-setup-04-run-density/
│       └── test-setup-04-run-density-yolo26-pose-count.ipynb
│
├── validation/
│   ├── 04-VALIDATION-README.md
│   ├── CONFIG-VALIDATION.yaml
│   └── test-quick-validation.ipynb
│
└── models/                    # Pre-trained YOLO26 models
    ├── yolo26n.pt            # Nano (5.3 MB)
    ├── yolo26m.pt            # Medium (42.2 MB)
    └── yolo26x.pt            # X-Large (113.2 MB)
```

**Note**: Model files are stored in `models/` directory. All data files are self-contained within each test suite.

## ⚙️ Configuration Files

Each directory has a `config.yaml` file with:

- **dependencies**: Required Python packages
- **hardware_requirements**: GPU/CPU specifications  
- **parameters**: Default values for hyperparameters
- **estimated_runtime_minutes**: Expected execution time
- **expected_outputs**: Output files generated

## 🎯 Key Features

✓ **Organized by function** - Clear purpose for each directory  
✓ **Well-documented** - README + config + docstrings  
✓ **Self-contained** - Each directory has everything needed  
✓ **Professional** - Production-ready structure  
✓ **Scalable** - Easy to add more notebooks  
✓ **Maintainable** - Clear organization and documentation  

## 🆘 Need Help?

1. **Setup issues** → Read: `setup/01-SETUP-README.md`
2. **Training issues** → Read: `training/02-TRAINING-README.md`
3. **Testing issues** → Read: `testing/03-TESTING-README.md`
4. **Validation issues** → Read: `validation/04-VALIDATION-README.md`

Each README includes a troubleshooting section!

## 📊 Key Information

| Aspect | Details |
|--------|---------|
| **Total Notebooks** | 10 |
| **Categories** | 4 (setup, training, testing, validation) |
| **Documentation Files** | 8 (4 READMEs + 4 CONFIG files) |
| **Setup Time** | 15 minutes |
| **Total Runtime** | ~2.5 hours (end-to-end) |
| **GPU Required** | Recommended (CPU fallback available) |
| **Min VRAM** | 4GB |
| **Recommended VRAM** | 8GB+ |

## 🔗 Parent Repository

This notebook collection is part of: **count-github-yolo-01**  
Main README: `../README.md`  
Main Dependencies: `../requirements.txt`

---

**Status**: ✅ Fully organized and documented  
**Last Updated**: 2026-05-26  
**Total Notebooks**: 10  
**Categories**: 4
