# Troubleshooting: OpenMP Kernel Crash

## Problem

When running Jupyter notebooks, you see this error:

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
The Kernel crashed while executing code in the current cell...
```

## Root Cause

Multiple Python packages (NumPy, SciPy, Pandas, scikit-learn, etc.) are compiled with Intel MKL (Math Kernel Library), which includes its own OpenMP runtime. When these packages are imported in the same Python process, the OpenMP libraries conflict.

## Solutions

### Solution 1: Quick Fix (Temporary - Works Now)

The notebook already includes this fix. In the first cell, we set:

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
```

**Run this cell first** before running other cells.

**Pros**: 
- Immediate fix
- No reinstallation needed
- Works within the notebook session

**Cons**:
- Only works for current notebook session
- Not recommended for production
- May cause performance issues or incorrect results

### Solution 2: Environment Variable (Persistent for Session)

Set the environment variable before starting Python/Jupyter:

**Windows (Command Prompt):**
```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
jupyter notebook
```

**Windows (PowerShell):**
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
jupyter notebook
```

**Windows (.env file in project root):**
```
KMP_DUPLICATE_LIB_OK=TRUE
```

### Solution 3: Proper Fix (Remove MKL, Use OpenBLAS)

This is the recommended permanent fix.

**Step 1: Identify which packages have MKL**
```bash
python -c "import numpy; import sys; print(numpy.__config__.show())" | grep blas
```

**Step 2: Uninstall MKL-linked packages**
```bash
pip uninstall numpy scipy scikit-learn -y
```

**Step 3: Reinstall without MKL (using conda or pip with nomkl)**

Using conda (recommended):
```bash
conda install numpy scipy scikit-learn nomkl
conda remove mkl mkl-service
```

Using pip:
```bash
pip install numpy scipy scikit-learn --no-cache-dir
```

**Step 4: Verify installation**
```bash
python -c "import numpy; print(numpy.show_config())"
# Should show: OpenBLAS or BLIS (not MKL)
```

### Solution 4: Use OpenBLAS Explicitly

**Windows - via vcpkg or manual installation:**

1. Download OpenBLAS from: https://github.com/xianyi/OpenBLAS/releases
2. Extract and add to PATH
3. Reinstall NumPy pointing to OpenBLAS:

```bash
pip install --no-cache-dir numpy
```

## Affected Notebooks

This issue affects validation notebooks that import numerical libraries:
- `notebooks/validation/test-quick-validation.ipynb` (FIXED)
- `notebooks/setup/test-setup-01.ipynb`
- `notebooks/training/train-*.ipynb`
- `notebooks/testing/test-setup-*.ipynb`

## For Each Notebook

Add this cell as the **first executable cell**:

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print("OpenMP fix applied: KMP_DUPLICATE_LIB_OK=True")
```

## Technical Details

**What's happening:**

1. Package A links: libiomp5md.dll (OpenMP)
2. Package B also links: libiomp5md.dll
3. When both load, they conflict
4. Kernel crashes with Exit Code 3

**Why it happens on Windows:**

- Intel MKL is Windows default for numerical packages
- Multiple copies get linked statically
- Dynamic linking would prevent this, but MKL uses static by default

## Prevention

Use environments with consistent BLAS/LAPACK:

```bash
# Create a fresh environment
python -m venv venv
.\venv\Scripts\activate

# Install with nomkl constraint
pip install nomkl
pip install numpy scipy scikit-learn pandas
```

## References

- [NumPy + MKL issues](https://github.com/numpy/numpy/issues)
- [Intel OpenMP Documentation](https://www.intel.com/content/www/en/en/developer/articles/technical/avoid-omp-error.html)
- [Conda MKL removal](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)

## When to Use Each Solution

| Solution | When to Use | Duration |
|----------|-----------|----------|
| Solution 1 | Quick testing, development | Current session only |
| Solution 2 | Regular use, all notebooks | Current console session |
| Solution 3 | Production, CI/CD pipelines | Permanent (best) |
| Solution 4 | Specific requirements | Permanent, high control |

---

**Recommended**: Use **Solution 3** for permanent fix. Use **Solution 1** as immediate workaround while setting up Solution 3.

**Current status**: test-quick-validation.ipynb includes Solution 1 as first cell.
