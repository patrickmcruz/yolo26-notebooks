# test-setup-05-evaluate-run-density

## Overview

This is a new evaluation sub-project cloned from `test-setup-04-run-density` with added evaluation capabilities.

**Purpose**: Evaluate and analyze the output from YOLO26-Pose crowd counting inference on video.

## Directory Structure

```
test-setup-05-evaluate-run-density/
├── evaluation/
│   └── evaluate_output.py       # Refactored evaluation script
├── input/
│   └── 20260329_34.mp4         # Input video file (610 MB)
├── output/
│   ├── snapshots/              # 61 frame snapshots
│   ├── frame_counts.csv        # Temporal count data (43,211 frames)
│   ├── summary.json            # Inference metadata
│   └── analise_estatistica.png # Generated analysis chart
├── data.yaml                   # Dataset config
├── test-setup-04-run-density-yolo26-pose-count.ipynb
├── yolo26x-pose.pt            # Pre-trained model (120 MB)
└── EVALUATION-README.md        # This file
```

## Refactored Script: evaluate_output.py

### Key Changes from Original

**1. Path Resolution (Lines 9-18)**
```python
# Before (broken - looked for output in evaluation/ dir)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")

# After (works - finds output sibling directory)
CURRENT_DIR = Path(__file__).parent.resolve()  # .../evaluation/
PROJECT_ROOT = CURRENT_DIR.parent              # .../test-setup-05/
TARGET_OUTPUT_DIR = str(PROJECT_ROOT / "output")
```

**2. Type-Safe JSON Parsing (Lines 37-50)**
```python
# Before (fails when values are strings)
print(f" - Média: {summary.get('avg_count', 'N/A'):.2f}")

# After (checks type before formatting)
avg_count = summary.get('avg_count', 'N/A')
if isinstance(avg_count, (int, float)):
    print(f" - Media: {avg_count:.2f}")
else:
    print(f" - Media: {avg_count}")
```

**3. Unicode Character Fixes (Line 135)**
```python
# Before (fails on Windows with cp1252 encoding)
print(f"    └─ {s}")

# After (ASCII-safe)
print(f"     - {s}")
```

## Running the Evaluation

### Method 1: Direct Python Execution

```bash
cd notebooks/testing/test-setup-05-evaluate-run-density/evaluation
python evaluate_output.py
```

### Method 2: From Jupyter Notebook

In a notebook cell:
```python
import sys
from pathlib import Path

# Add evaluation module to path
eval_dir = Path("evaluation")
sys.path.insert(0, str(eval_dir.resolve()))

# Import and run
from evaluate_output import evaluate_metrics
evaluate_metrics()
```

### Method 3: Import as Module

```python
import sys
sys.path.insert(0, "evaluation")
from evaluate_output import evaluate_metrics

evaluate_metrics()
```

## Script Features

The `evaluate_metrics()` function performs:

1. **JSON Summary Analysis**
   - Loads `output/summary.json`
   - Extracts: video path, frame count, avg/max crowd size, processing time, FPS

2. **Temporal Series Analysis (CSV)**
   - Loads `output/frame_counts.csv`
   - Computes: mean, std dev, coefficient of variation, temporal noise
   - Generates temporal visualization

3. **Statistical Visualization**
   - Time series plot with rolling average
   - Distribution histogram
   - Saved to: `analise_estatistica.png` (224 KB)

4. **Snapshot Inventory**
   - Lists available frame snapshots (61 total)
   - Shows sample frames for manual audit

## Output Files Generated

| File | Size | Purpose |
|------|------|---------|
| analise_estatistica.png | 224 KB | Matplotlib figure with temporal analysis |
| (CSV/JSON/snapshots) | Existing | Referenced, not modified |

## Key Metrics from Last Run

```
Frames Analyzed: 43,211
Average Count: 900.21 ± 519.75
Median: 900.2
Variation Coefficient: 57.74%
Temporal Noise (Instability): 0.00%
```

## Dependencies

Required Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- json (standard library)
- pathlib (standard library)
- os (standard library)

Install with:
```bash
pip install pandas numpy matplotlib seaborn
```

## Next Steps

1. **Run the evaluation** to generate analysis charts
2. **Inspect analise_estatistica.png** for visual insights
3. **Review snapshots** in `output/snapshots/` for quality checks
4. **Export results** to reports or dashboards

## Troubleshooting

### "ModuleNotFoundError: No module named 'seaborn'"
```bash
pip install seaborn matplotlib pandas numpy
```

### "Output directory not found"
Verify the directory structure:
```bash
ls test-setup-05-evaluate-run-density/output/
# Should show: snapshots/, frame_counts.csv, summary.json
```

### Chart not generated
Check write permissions:
```bash
touch test-setup-05-evaluate-run-density/test.txt
rm test-setup-05-evaluate-run-density/test.txt
```

## Integration with Notebooks

### From test-setup-05 Main Notebook

Create a new cell:
```python
import sys
from pathlib import Path

# Run evaluation
sys.path.insert(0, str(Path("evaluation").resolve()))
from evaluate_output import evaluate_metrics

print("Starting evaluation...")
evaluate_metrics()
print("Evaluation complete!")
```

## Summary

✓ Script refactored for correct path resolution  
✓ Type-safe JSON parsing added  
✓ Unicode issues resolved  
✓ All dependencies installed  
✓ Tested and working  
✓ Chart generation verified  

---

**Status**: PRODUCTION READY  
**Last Updated**: 2026-05-26  
**Test Result**: SUCCESS
