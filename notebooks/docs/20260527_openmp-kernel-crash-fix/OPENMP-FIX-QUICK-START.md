# OpenMP Fix - Quick Start Guide

## TL;DR

If your Jupyter notebook crashes with **"libiomp5md.dll already initialized"**:

1. Open the notebook
2. Run the **first cell** (it contains the OpenMP fix)
3. Continue with other cells normally

## What Was Fixed?

✅ `notebooks/validation/test-quick-validation.ipynb`  
✅ `notebooks/setup/test-setup-01.ipynb`  
✅ Created comprehensive troubleshooting guide

## For Your Next Jupyter Session

When opening any notebook:

1. **Look for the "Fix: OpenMP Conflict" cell** at the beginning
2. **Run it first** (before any other cells)
3. Then proceed normally

The fix cell looks like this:
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print("OpenMP conflict mitigation: ENABLED")
```

## Why This Happens?

Your system has multiple Python packages linked to Intel's OpenMP library. When they load together, they conflict. This fix tells Intel to ignore the duplication.

## Need More Help?

See **TROUBLESHOOTING-OpenMP.md** for:
- 4 different solutions
- How to apply a permanent fix
- Technical details
- FAQ

## Affected Notebooks

Currently fixed:
- ✅ `notebooks/validation/test-quick-validation.ipynb`
- ✅ `notebooks/setup/test-setup-01.ipynb`

May also need the fix:
- `notebooks/training/train-*.ipynb`
- `notebooks/testing/test-setup-*.ipynb`

## One-Time Permanent Fix

If you want to fix this permanently (recommended):

**Option A: Use Conda (Easiest)**
```bash
conda install nomkl
conda remove mkl mkl-service -y
```

**Option B: Use pip**
```bash
pip uninstall numpy scipy scikit-learn -y
pip install numpy scipy scikit-learn
```

Then restart Jupyter and run notebooks normally without the fix cell.

---

**Still having issues?** Check TROUBLESHOOTING-OpenMP.md for detailed solutions.
