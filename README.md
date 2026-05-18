## YOLO26 Custom Training Notebook

This repository now focuses on a single, practical workflow notebook:

- [notebooks/test-setup-02-train-hparams/test-setup-02-train-hparams.ipynb](notebooks/test-setup-02-train-hparams/test-setup-02-train-hparams.ipynb)

The notebook is structured to be modular, restart-safe, and easy to reason about on Windows. It uses centralized configuration, a managed folder layout, a fast validation preview, and two training profiles: a smoke test and a light scale-up branch.

## Notebook Guide

This section explains the notebook in the same order the cells are meant to run.

### 1. Title and DAG overview

Cells 1 to 2 introduce the notebook and show the workflow DAG.

- Cell 1 sets the purpose of the notebook: a compact YOLO26 custom training workflow.
- Cell 2 shows the flow from environment setup to dataset download, preflight checks, smoke training, and optional scale-up training.
- The DAG is useful because it makes the notebook control flow visible before you run anything.

### 2. GPU preflight

Cell 3 explains the first gate, and Cell 4 checks `nvidia-smi`.

- The notebook fails fast if the GPU is not visible.
- This is the cheapest way to avoid spending time on package installation or dataset setup when the machine is not ready.

### 3. Centralized workspace layout

Cell 5 describes the workspace strategy, and Cell 6 builds the folder map.

- `WORKSPACE_ROOT` and `NOTEBOOKS_ROOT` are resolved once and reused everywhere.
- The notebook creates dedicated directories for datasets, results, models, logs, and artifacts.
- This keeps training output organized and reduces the chance of scattering files across the workspace.

### 4. Dependency installation

Cell 7 installs the runtime dependencies.

- PyTorch is installed with CUDA 12.4 wheels.
- Ultralytics, Roboflow, and the supporting packages are installed together.
- Ultralytics telemetry is disabled with `yolo settings sync=False` so the notebook does not require external syncing.

### 5. Dataset bootstrap

Cells 8 to 10 prepare the dataset.

- Cell 8 marks the start of the custom training section.
- Cell 9 reminds you that the dataset should live under `datasets/` and be exported in YOLO format.
- Cell 10 prints the managed directories for confirmation.

Cell 11 loads the Roboflow client and handles the HTTPS verification workaround used in this environment.

- The notebook loads the API key from `.env`.
- The Roboflow workspace, project, and version are centralized as constants.
- The dataset is downloaded into the managed `datasets/` tree.

Cell 12 performs the actual download.

- The notebook expects the YOLO26 export for `basketball-player-detection-3-14`.
- The generated dataset path is stored in `dataset_path` and `data_yaml` is derived from it.

### 6. Fast validation preview

Cell 13 introduces the preflight stage, and Cell 14 builds a fast validation preview YAML.

- The notebook copies only a handful of validation images into a short helper path.
- It writes `data.fast.yaml` so Ultralytics can train against a tiny validation target instead of scanning the full validation split.
- This is a practical workaround for long `val: Scanning` phases and path-length issues on Windows.

### 7. Model preflight

Cell 15 loads the YOLO model and runs model-level preflight checks.

- The notebook checks CUDA availability, model weights, and the dataset YAML before training.
- It also restores the model path in the Ultralytics overrides so repeated notebook runs stay stable.

### 8. Runtime diagnostics

Cell 16 prints runtime diagnostics.

- Python version and executable path.
- Platform details.
- Torch and Ultralytics versions.
- CUDA device count, device name, and memory.
- Dataset file counts and samples.

This cell is intentionally verbose because training failures are easier to debug when the environment state is printed in one place.

### 9. Smoke profile

Cells 17 and 18 define and run the smoke profile.

- Cell 17 rebuilds the model and creates the smoke training arguments.
- Cell 18 launches the smoke run.
- The smoke run is meant to be the smallest useful training execution, not a serious training attempt.

### 10. Scale-up branch

Cells 19 to 22 define the optional scale-up path.

- Cell 20 builds the scale-up training arguments.
- Cell 21 checks that the model override still contains the model path.
- Cell 22 runs the scale-up training branch.

The scale-up branch is optional. It is useful after the smoke run has already proven that the notebook, dataset, and GPU can work together.

## Training Chapter

This notebook uses two training profiles. Both are built from the same base config, both use the same dataset, and both force `workers = 0` for Windows reliability.

### Shared design choices

The notebook keeps the following decisions centralized:

- All paths are defined in one config block.
- All training profiles inherit the same base `project` and `exist_ok` settings.
- The model is always loaded from `yolo26n.pt`.
- The notebook writes outputs into the managed workspace tree instead of the project root.
- Validation uses a preview YAML so the notebook does not spend time on a full scan every time.

### Smoke profile details

The smoke profile is the first training branch. Its purpose is to prove that the full workflow can start and complete with the smallest reasonable cost.

| Hyperparameter | Value | Reason |
|---|---:|---|
| `epochs` | `1` | One epoch is enough to validate the pipeline without turning the notebook into a long job. |
| `imgsz` | `320` | Smaller images reduce memory pressure and start faster, which is ideal for a smoke test. |
| `batch` | `1` | Batch size one is the safest choice for a quick reliability check. |
| `device` | `0` | Uses the first visible CUDA device directly. |
| `workers` | `0` | Avoids multiprocessing dataloader crashes on Windows. |
| `cache` | `False` | Keeps the smoke run light and avoids extra disk work. |
| `pretrained` | `True` | Starts from pretrained weights so the test focuses on workflow health. |
| `optimizer` | `SGD` | A conservative optimizer that is easy to reason about for a minimal pass. |
| `lr0` | `0.001` | Low initial learning rate reduces the chance of unstable updates. |
| `cos_lr` | `False` | Keeps the smoke profile simple and predictable. |
| `patience` | `1` | Fails fast if the training loop is not behaving as expected. |
| `fraction` | `0.01` | Uses only a tiny share of the dataset so the run is fast. |
| `mosaic` | `0.0` | Removes heavy augmentation from the first pass. |
| `mixup` | `0.0` | Keeps the smoke run deterministic and simple. |
| `fliplr` | `0.0` | Avoids extra augmentation noise in the first test. |
| `close_mosaic` | `1` | Keeps mosaic effectively disabled for the whole smoke branch. |
| `plots` | `False` | Reduces overhead during the quick validation path. |
| `amp` | `True` | Uses mixed precision when supported to reduce memory pressure. |
| `verbose` | `True` | Makes notebook logs explicit and easier to debug. |
| `save` | `False` | Avoids unnecessary artifact generation during the smoke run. |
| `val` | `False` | Prevents a full validation pass after training. |

### Scale-up profile details

The scale-up profile is still intentionally light, but it raises the load enough to catch problems that the smoke profile might miss.

| Hyperparameter | Value | Reason |
|---|---:|---|
| `epochs` | `3` | Long enough to exercise the training loop a bit more without becoming expensive. |
| `imgsz` | `480` | A moderate increase in image size adds realism and memory pressure. |
| `batch` | `1` | Keeps GPU memory use predictable. |
| `device` | `0` | Uses the first CUDA device consistently. |
| `workers` | `0` | Keeps the notebook stable on Windows. |
| `cache` | `False` | Keeps the branch lean and avoids cache churn. |
| `pretrained` | `True` | Fine-tuning should start from pretrained weights. |
| `optimizer` | `AdamW` | Better suited for a slightly longer fine-tuning pass than the smoke baseline. |
| `lr0` | `0.002` | Slightly higher to make the short scale-up branch more productive. |
| `cos_lr` | `True` | Adds a more realistic learning-rate schedule. |
| `patience` | `2` | Gives the branch a little more room before stopping early. |
| `fraction` | `0.05` | Uses more data than the smoke run while still staying quick. |
| `mosaic` | `0.25` | Introduces modest augmentation for a more realistic training pass. |
| `mixup` | `0.0` | Left off because it adds complexity that is not needed here. |
| `fliplr` | `0.1` | Adds a small amount of horizontal flipping for generalization. |
| `close_mosaic` | `1` | Prevents mosaic from dominating the short run. |
| `plots` | `True` | Makes it easier to inspect the run if you choose to execute this branch. |
| `amp` | `True` | Keeps GPU memory use lower and training faster where supported. |
| `verbose` | `True` | Surfaces detailed logs for inspection. |
| `save` | `True` | Saves outputs so the scale-up branch leaves artifacts behind. |
| `val` | `False` | Avoids a full end-of-training validation pass because the notebook already uses the preview split. |

### Why this notebook avoids full validation scans

Ultralytics can spend time scanning the validation split before training starts. For a notebook workflow, that makes the first run feel slow and fragile. The fast preview YAML keeps validation small, predictable, and easy to rerun.

### Why `workers = 0` is the default

On Windows, notebook dataloader workers are a common source of crashes. Setting `workers = 0` keeps loading in the main process and makes the workflow much more reliable.

## Recommended run order

1. Run the GPU check cell.
2. Run the workspace config cell.
3. Install dependencies.
4. Download the dataset.
5. Build the fast validation preview.
6. Run preflight and runtime diagnostics.
7. Execute the smoke profile.
8. Only then try the scale-up branch.

## Output layout

The notebook keeps its files in a managed structure:

- `datasets/` for downloaded data.
- `results/` for Ultralytics training output.
- `models/` for model artifacts.
- `logs/` for runtime logs.
- `artifacts/` or `_v/` for helper outputs such as the fast validation preview.

