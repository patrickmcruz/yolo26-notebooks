# For Users: How to Fix the Kernel Crash

## The Problem You See

```
Error: OMP: Error #15: Initializing libiomp5md.dll already initialized.
[error] Disposing session as kernel process died ExitCode: 3
```

## The Solution (2 Steps)

### Step 1: Open Your Notebook

For example: `notebooks/validation/test-quick-validation.ipynb`

### Step 2: Run This Cell First

Look for a cell at the beginning that looks like this:

```python
import os

# Fix for: OMP: Error #15: libiomp5md.dll already initialized
# This occurs when multiple packages link Intel OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print("OpenMP conflict mitigation: ENABLED")
print("Set KMP_DUPLICATE_LIB_OK=True")
print()
print("Note: This is a temporary workaround.")
print("For a permanent fix, see the TROUBLESHOOTING-OpenMP.md file.")
```

**Just run it!** (Press Shift+Enter or click Run)

You should see output like:
```
OpenMP conflict mitigation: ENABLED
Set KMP_DUPLICATE_LIB_OK=True

Note: This is a temporary workaround.
For a permanent fix, see the TROUBLESHOOTING-OpenMP.md file.
```

### Step 3: Run Rest of Notebook Normally

After running that first cell, all other cells should work without crashing!

---

## Affected Notebooks

These notebooks now have the fix included:
- ✅ `notebooks/validation/test-quick-validation.ipynb` 
- ✅ `notebooks/setup/test-setup-01.ipynb`

## Still Getting Crash?

1. **Restart the kernel** first
2. **Run the fix cell immediately** (before any other cells)
3. Then run your other cells

---

## Want a Permanent Fix?

If you want to fix this permanently (so you don't need the cell every time):

See: `notebooks/TROUBLESHOOTING-OpenMP.md` → **Solutions Section**

Quick option:
```bash
# If you use conda
conda install nomkl
conda remove mkl mkl-service -y
```

Then restart Jupyter and run notebooks normally.

---

## That's It!

The kernel crash is now fixed. Just remember to run the first cell before running other cells in those notebooks.

If you see the fix cell in your notebook, run it first. Then proceed normally.

**Questions?** Check `notebooks/OPENMP-FIX-QUICK-START.md`
