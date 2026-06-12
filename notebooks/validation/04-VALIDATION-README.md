# 04-validation: Quick Sanity Checks & End-to-End Validation

## Overview

This directory contains quick validation and sanity check notebooks for end-to-end pipeline validation.

**Purpose**: Fast feedback on model inference without extensive testing.

**Speed**: < 1 minute per notebook

**GPU Required**: No (can run on CPU)

---

## Notebook

### `test-quick-validation.ipynb`

**Purpose**: Quick end-to-end validation of complete inference pipeline.

**What it does**:

1. **Model Loading**:
   - Loads pre-trained YOLO model
   - Verifies model integrity
   - Checks weights integrity

2. **Sample Images**:
   - Uses test dog images (dog-2.jpeg, dog-3.jpeg)
   - Downloads images if not available
   - Validates image format

3. **Inference**:
   - Runs inference on sample images
   - Measures inference speed (FPS)
   - Detects objects (people, dogs, etc.)
   - Estimates poses (if available)

4. **Output Generation**:
   - Draws bounding boxes on images
   - Marks pose keypoints
   - Saves annotated images
   - Generates results summary

5. **Validation Checks**:
   - ✅ Model loads successfully
   - ✅ Inference produces detections
   - ✅ Output format is correct
   - ✅ Inference speed acceptable

### Configuration

```yaml
category: validation
description: Quick end-to-end validation

model:
  name: yolo26m.pt
  type: object_detection
  size: medium

test_images:
  - dog-2.jpeg
  - dog-3.jpeg
  - count: 2-5 images

inference_params:
  confidence_threshold: 0.5
  iou_threshold: 0.45

outputs:
  - Annotated images
  - Detection summary
  - Inference metrics

hardware_requirements:
  min_vram_gb: 2
  gpu_required: false  # CPU capable
  
estimated_runtime_minutes: 1
```

### Expected Output

```
Quick Validation Results
========================

Model: yolo26m.pt
Device: GPU (CUDA)
Inference Speed: 45 FPS (GPU) or 2 FPS (CPU)

Image: dog-2.jpeg
  Detections: 1 dog
  Confidence: 0.95
  Detection time: 22ms

Image: dog-3.jpeg
  Detections: 2 dogs
  Confidence: 0.91, 0.88
  Detection time: 24ms

✅ Validation PASSED
- Model loaded successfully
- Inference works correctly
- Output format valid
- Performance acceptable
```

### Success Criteria

All of these should pass:
- ✅ Model file found and loads
- ✅ Sample images available
- ✅ Inference completes without errors
- ✅ Detections with reasonable confidence
- ✅ Output images generated
- ✅ Inference speed > 1 FPS

### Troubleshooting

#### Problem: "Model file not found"
**Solution**:
```bash
# Download model weights
python -c "from ultralytics import YOLO; YOLO('yolo26m.pt')"
```

#### Problem: "No detections in images"
**Solution**:
- Lower confidence threshold: 0.3 instead of 0.5
- Try different model: yolo26x instead of yolo26m
- Verify sample images exist

#### Problem: Slow inference (< 1 FPS)
**Solution**:
- Use GPU instead of CPU
- Try smaller model variant
- Check GPU is properly utilized

#### Problem: Import errors
**Solution**:
```bash
pip install -q ultralytics torch torchvision
```

#### Problem: "Kernel crashed" / "libiomp5md.dll already initialized"
**Solution**:
This is an OpenMP conflict issue. The notebook has a fix:
1. **Run the first cell** in the notebook (it contains the OpenMP fix)
2. Then proceed with other cells
3. For permanent fix, see: `notebooks/TROUBLESHOOTING-OpenMP.md`

The first cell contains:
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
```

This tells Python to ignore the duplicate OpenMP runtime initialization.

---

## Quick Validation Workflow

### Pre-Deployment Checklist (2 minutes)

```
1. Run test-quick-validation.ipynb
   ↓ Check all outputs are generated
   ↓ Verify inference speed acceptable
   ↓ Confirm detection quality looks good

2. Review results:
   - Check annotated images
   - Verify confidence scores reasonable
   - Confirm no import errors

3. Decision:
   ✅ PASS → Ready for deployment
   ❌ FAIL → Debug issues before deployment
```

### CI/CD Integration

Use this notebook for continuous validation:

```bash
# Automated pre-deployment check
jupyter nbconvert --to notebook --execute test-quick-validation.ipynb \
  --output-format ipynb --output quick-validation-results.ipynb
```

---

## Configuration File

**File**: `config.yaml`

```yaml
category: validation
type: quick-sanity-check
description: End-to-end pipeline validation

model_config:
  model_path: ../02-training/runs/detect/train/weights/best.pt
  backup_model: yolo26m.pt
  
inference_config:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: auto  # auto-detect GPU/CPU
  
test_config:
  test_images:
    - dog-2.jpeg
    - dog-3.jpeg
  max_images: 10
  
expected_metrics:
  min_fps: 1  # minimum acceptable FPS
  max_inference_time_ms: 500
  acceptable_confidence: 0.3  # minimum acceptable confidence

hardware_requirements:
  min_vram_gb: 2
  gpu_required: false
  can_run_on_cpu: true

estimated_runtime_minutes: 1
```

---

## Use Cases

### Use Case 1: Pre-Deployment Validation
**When**: Before deploying model to production
**Action**: Run this notebook to ensure everything works
**Decision**: Go/no-go for deployment

### Use Case 2: Regression Testing
**When**: After model updates or dependency changes
**Action**: Run to ensure no breakage
**Decision**: Is model still working after changes?

### Use Case 3: CI/CD Pipeline
**When**: Automated testing on every commit
**Action**: Auto-run after model updates
**Decision**: Fail build if validation fails

### Use Case 4: Demo/Verification
**When**: Showing model to stakeholders
**Action**: Run notebook to show working inference
**Decision**: Quality check before presentation

---

## Integration with Other Notebooks

### Workflow: Setup → Training → Validation

```
01-setup/
├── setup-density-hardware-analysis.ipynb
└── test-setup-01.ipynb
        ↓
        ↓ Validates environment
        ↓
02-training/
├── train-yolo26-object-detection-on-custom-dataset.ipynb
└── (produces runs/detect/train/weights/best.pt)
        ↓
        ↓ Trains model
        ↓
04-validation/
└── test-quick-validation.ipynb ← YOU ARE HERE
        ↓
        ↓ Validates inference works
        ↓
03-testing/ (optional, for detailed testing)
├── test-setup-02-train-hparams/
├── test-setup-03-run-few/
└── test-setup-04-run-density/
```

### Model Loading Hierarchy

The notebook searches for models in this order:
1. `../02-training/runs/detect/train/weights/best.pt` (last trained)
2. `yolo26m.pt` (pre-trained backup)
3. Download from Ultralytics if not found

---

## Performance Baselines

### Expected Inference Speed

| Model | GPU (RTX 3090) | GPU (RTX 3070) | CPU |
|-------|---|---|---|
| yolo26n | 150 FPS | 80 FPS | 5 FPS |
| yolo26m | 100 FPS | 45 FPS | 2 FPS |
| yolo26l | 60 FPS | 25 FPS | 1 FPS |
| yolo26x | 40 FPS | 15 FPS | 0.5 FPS |

### Expected Detection Confidence

- Well-trained model: 0.85-0.98
- Acceptable range: 0.5-0.95
- If lower: Model may need more training

---

## Output Files

After running the notebook:

```
runs/detect/predict/
├── dog-2.jpg (annotated output)
├── dog-3.jpg (annotated output)
└── results.txt (detection summary)
```

Also generates in notebook output:
- Inference time metrics
- FPS measurement
- Confidence statistics
- Detection summary

---

## Next Steps

### If Validation Passes ✅
1. Model is ready for deployment
2. Proceed to integration testing
3. Deploy to production
4. Document model card

### If Validation Fails ❌
1. Check error messages carefully
2. Verify prerequisites installed
3. Review setup notebooks (01-setup/)
4. Re-run training if needed (02-training/)
5. Try again

### For Detailed Testing
1. Run few-shot tests: See `03-testing/test-setup-03-run-few/`
2. Run density tests: See `03-testing/test-setup-04-run-density/`
3. Optimize hyperparameters: See `03-testing/test-setup-02-train-hparams/`

---

## Model Card Template

When validation passes, document your model:

```markdown
# Model Card: YOLO26 Object Detection

## Model Details
- **Model Type**: YOLOv8 Object Detection
- **Variant**: yolo26m.pt
- **Input Size**: 960x960
- **Training Epochs**: 50

## Performance
- **Inference Speed**: 45 FPS (GPU), 2 FPS (CPU)
- **Detection Accuracy**: mAP=0.91
- **Confidence Threshold**: 0.5

## Validation
- **Quick Validation**: ✅ PASSED
- **Last Validated**: 2026-05-26
- **Validated By**: Automated pipeline

## Use Cases
- Object detection in crowded scenes
- Person counting
- Real-time inference

## Limitations
- Performance degrades in very high-density scenarios (>50 objects)
- Requires GPU for real-time inference (>20 FPS)
```

---

**Status**: ✅ Complete  
**Last Updated**: 2026-05-26  
**Runtime**: ~1 minute  
**GPU Required**: No (can run on CPU)  
**Previous**: See `03-testing/README.md`
