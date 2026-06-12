# 02-training: Model Training Workflows

## Overview

This directory contains complete training pipelines for various computer vision tasks using YOLO26 and related models.

**Prerequisites**: Complete all notebooks in `01-setup/` directory first.

## Notebooks

### 1. `_train-template.ipynb` (Reference Template)

**Purpose**: Provides a reusable template for creating task-specific training workflows.

**Structure**:
- Step 0: GPU/environment check
- Step 1: Clone repository and install dependencies
- Step 2: Test inference with pre-trained model
- Step 3: Prepare custom dataset
- Step 4: Training loop with metrics
- Step 5: Evaluation and visualization
- Step 6: Active learning (optional)

**Use case**: Reference when creating new training notebooks

**Not meant to be run directly** (contains placeholders for customization)

---

### 2. `train-yolo26-object-detection-on-custom-dataset.ipynb`

**Purpose**: Complete training pipeline for object detection (bounding box localization).

**What it does**:
1. **Setup**: Installs dependencies, verifies GPU
2. **Data Preparation**: 
   - Downloads dataset from Roboflow
   - Validates dataset format
   - Splits into train/val/test
3. **Model Training**:
   - Initializes YOLO26 model
   - Trains with configured hyperparameters
   - Monitors loss and metrics
4. **Evaluation**:
   - Computes metrics (mAP, precision, recall)
   - Generates confusion matrices
   - Visualizes results
5. **Inference**:
   - Tests on validation images
   - Generates annotated predictions
   - Saves best model

**Task Definition**:
- **Input**: Images with bounding box annotations
- **Output**: Detected objects with bounding boxes + confidence scores
- **Format**: YOLO format (object categories with box coordinates)

**Prerequisites**:
- Setup validation complete (01-setup/)
- Custom dataset prepared (from Roboflow or local)
- 8GB+ GPU VRAM
- ~30GB disk space for model and results

**Hyperparameters**:
- **Batch size**: 16 (adjust based on GPU memory)
- **Image size**: 960 (from hardware analysis)
- **Epochs**: 50 (can increase for better accuracy)
- **Learning rate**: 0.001
- **Optimizer**: SGD with momentum

**Expected outputs**:
- Trained model: `runs/detect/train*/weights/best.pt`
- Training metrics: `runs/detect/train*/results.csv`
- Predictions: `runs/detect/train*/predictions/`
- Confusion matrix: `runs/detect/train*/confusion_matrix.png`

**Runtime**: 30-60 minutes (varies with dataset size and GPU)

**Performance expectations**:
- mAP50: 0.85-0.95 (object detection baseline)
- Training loss: Should decrease smoothly
- Validation loss: Should stabilize

---

### 3. `train-yolo26-instance-segmentation-on-custom-dataset.ipynb`

**Purpose**: Complete training pipeline for instance segmentation (pixel-level masks).

**What it does**:
1. **Setup**: Validates environment and GPU
2. **Data Preparation**:
   - Downloads segmentation dataset
   - Validates mask formats
   - Converts to YOLO segmentation format
3. **Model Training**:
   - Initializes YOLO26-seg model
   - Trains with mask loss + detection loss
   - Monitors multi-task learning
4. **Evaluation**:
   - Computes metrics (mAP, mask IoU)
   - Visualizes segmentation masks
   - Generates quality reports
5. **Inference**:
   - Tests on validation images
   - Generates annotated masks
   - Saves best model

**Task Definition**:
- **Input**: Images with instance mask annotations
- **Output**: Detected objects + pixel-level segmentation masks
- **Format**: YOLO segmentation format (box + polygon masks)

**Prerequisites**:
- Setup validation complete (01-setup/)
- Segmentation dataset with masks
- 12GB+ GPU VRAM recommended
- ~40GB disk space

**Hyperparameters** (adjusted for segmentation):
- **Batch size**: 8 (masks require more memory)
- **Image size**: 960
- **Epochs**: 75 (segmentation needs more training)
- **Learning rate**: 0.001
- **Focus loss**: Enabled for hard example mining

**Expected outputs**:
- Trained model: `runs/segment/train*/weights/best.pt`
- Metrics: `runs/segment/train*/results.csv`
- Mask predictions: `runs/segment/train*/predictions/`
- Mask visualization: `runs/segment/train*/masks/`

**Runtime**: 45-90 minutes

**Performance expectations**:
- Box mAP: 0.80-0.92
- Mask IoU: 0.75-0.85
- Training time: 2-3x longer than detection

---

### 4. `how-to-finetune-rf-detr-on-segmentation-dataset-a100.ipynb`

**Purpose**: Advanced fine-tuning using DETR (Detection Transformer) on A100 GPU.

**What it does**:
1. **Setup**: Initializes transformer environment
2. **Data Loading**: Loads segmentation data with proper transforms
3. **Model Loading**: Loads pre-trained DETR model
4. **Fine-tuning**: 
   - Unfreezes late layers
   - Uses mixed precision training
   - Implements gradient accumulation
5. **Evaluation**: Tests on custom dataset

**Task**: Instance segmentation using transformer architecture

**Prerequisites**:
- **GPU**: A100 or H100 (40GB+ VRAM)
- **Knowledge**: Advanced PyTorch, transformers
- **Dataset**: Large-scale segmentation dataset
- **Runtime**: ~60-120 minutes

**Why DETR?**
- End-to-end transformer architecture
- Better handling of complex scenes
- State-of-the-art performance
- Trade-off: Slower inference, longer training

**Use cases**:
- When YOLO performance not sufficient
- Complex scenes with heavy occlusion
- When you have high-end GPU available
- Research/production deployment (accuracy > speed)

**Expected outputs**:
- Fine-tuned DETR model
- Evaluation metrics
- Segmentation masks

**Runtime**: 60-120 minutes on A100

---

## Configuration

**File**: `config.yaml`

```yaml
category: training
description: Model training workflows

dependencies:
  - ultralytics>=8.4.0
  - torch>=2.0.0
  - torchvision>=0.15.0
  - supervision
  - roboflow
  - numpy
  - pandas
  - opencv-python
  - pyyaml
  - matplotlib
  - scikit-learn

hardware_requirements:
  min_vram_gb: 8
  recommended_vram_gb: 16
  gpu_required: true
  cpu_cores_min: 4

parameters:
  object_detection:
    batch_size: 16
    image_size: 960
    epochs: 50
    learning_rate: 0.001
    optimizer: SGD
  
  instance_segmentation:
    batch_size: 8
    image_size: 960
    epochs: 75
    learning_rate: 0.001
    optimizer: SGD
  
  detr_finetuning:
    batch_size: 4
    image_size: 1024
    epochs: 30
    learning_rate: 0.00001
    optimizer: AdamW

estimated_runtime_minutes:
  object_detection: 60
  instance_segmentation: 90
  detr_finetuning: 120

expected_outputs:
  - runs/detect*/train*/weights/best.pt
  - runs/segment*/train*/weights/best.pt
  - runs/*/train*/results.csv
  - runs/*/train*/confusion_matrix.png
```

## Training Workflow

### Workflow 1: Object Detection Training (60 minutes)

```bash
1. Prepare dataset (from Roboflow or local)
   - Images in images/train, images/val, images/test
   - Annotations in labels/train, labels/val, labels/test
   - Format: YOLO txt format (class_id x_center y_center width height)

2. Run: train-yolo26-object-detection-on-custom-dataset.ipynb
   - Monitors training progress
   - Displays real-time metrics
   - Saves best checkpoint

3. Review outputs:
   - Best model: runs/detect/train*/weights/best.pt
   - Metrics: runs/detect/train*/results.csv
   - Predictions: runs/detect/train*/predictions/

4. Deploy:
   - Use best.pt for inference
   - Test on new images
   - Evaluate performance
```

### Workflow 2: Instance Segmentation Training (90 minutes)

```bash
1. Prepare segmentation dataset
   - Images and mask annotations
   - Convert masks to polygon format or binary masks
   - Split into train/val/test

2. Run: train-yolo26-instance-segmentation-on-custom-dataset.ipynb
   - Processes both box and mask data
   - Trains multi-task model
   - Validates on segmentation metrics

3. Review outputs:
   - Model: runs/segment/train*/weights/best.pt
   - Metrics: runs/segment/train*/results.csv
   - Mask visualizations

4. Deploy:
   - Use for pixel-level predictions
   - Post-process masks if needed
   - Integrate into pipeline
```

### Workflow 3: Advanced DETR Fine-tuning (120+ minutes)

```bash
1. Prepare large-scale segmentation dataset
2. Run: how-to-finetune-rf-detr-on-segmentation-dataset-a100.ipynb
   - On A100/H100 GPU (40GB+ VRAM)
   - Uses mixed precision for efficiency
   - Implements gradient accumulation
3. Evaluate and deploy
```

## Dataset Preparation

### Required Format: YOLO

**Object Detection**:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

**data.yaml**:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 1  # number of classes
names: ['person']  # class names
```

**Label format** (labels/train/*.txt):
```
<class_id> <x_center> <y_center> <width> <height>
0 0.5 0.5 0.8 0.9
```

### Using Roboflow

Most notebooks include Roboflow integration:

```python
# Download from Roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace").project("project-name")
dataset = project.versions(1).download("yolov8")
```

## Monitoring Training

During training, monitor:

1. **Training Loss**: Should decrease smoothly
2. **Validation Loss**: Should stabilize
3. **mAP (mean Average Precision)**: Should increase
4. **GPU Memory**: Should stay below max VRAM
5. **Training Speed**: FPS and time per epoch

**Good indicators**:
- ✅ Loss decreasing
- ✅ mAP increasing
- ✅ Validation loss following training loss
- ✅ No GPU out-of-memory errors

**Warning signs**:
- ⚠️ Loss increasing (learning rate too high)
- ⚠️ Loss not changing (learning rate too low)
- ⚠️ Validation loss >> training loss (overfitting)
- ⚠️ GPU memory full (reduce batch size)

## Hyperparameter Tuning

Start with provided values, then tune:

1. **Batch Size**: Larger = faster training but more memory
   - Reduce if: GPU out of memory
   - Increase if: GPU not fully utilized

2. **Learning Rate**: Controls training speed
   - Too high: Loss oscillates, doesn't converge
   - Too low: Training very slow
   - Start: 0.001, adjust ±10x

3. **Epochs**: Number of training passes
   - More epochs = better accuracy but longer training
   - Stop if validation loss plateaus

4. **Image Size**: Input resolution
   - Larger = better accuracy but slower, more memory
   - Common: 640, 960, 1024, 1280

For detailed hyperparameter search, see `03-testing/test-setup-02-train-hparams/`

## Troubleshooting

### Problem: "CUDA out of memory"
**Solution**:
- Reduce batch size (e.g., 16 → 8)
- Reduce image size (e.g., 960 → 640)
- Close other GPU applications

### Problem: Loss not decreasing
**Solution**:
- Increase learning rate (try 0.01)
- Check dataset is valid
- Verify label format is correct
- Try more epochs

### Problem: Training very slow
**Solution**:
- Increase batch size
- Reduce image size
- Enable mixed precision (FP16)
- Use larger GPU if available

### Problem: Model not converging
**Solution**:
- Check for data imbalance
- Verify augmentation is appropriate
- Try different learning rate schedule
- Add more training data

## Next Steps

After training:

1. **Validation**: 
   → Go to `04-validation/` for quick checks
   → Go to `03-testing/` for detailed testing

2. **Deployment**:
   → Export model to ONNX or TensorRT
   → Deploy to edge devices

3. **Optimization**:
   → Run `03-testing/test-setup-02-train-hparams/` to find better parameters
   → Re-train with optimal settings

---

**Status**: ✅ Complete  
**Last Updated**: 2026-05-26  
**Previous**: See `01-setup/README.md`  
**Next**: See `03-testing/README.md`
