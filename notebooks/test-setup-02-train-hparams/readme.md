# test-setup-02-train-hparams

This notebook trains a YOLO26 model on the Roboflow basketball player detection dataset. It is designed to run from inside its own folder and keeps generated artifacts next to the notebook so the project stays self-contained.

## Before You Run

1. Open the notebook from `notebooks/test-setup-02-train-hparams/test-setup-02-train-hparams.ipynb`.
2. Make sure `notebooks/.env` contains `ROBOFLOW_API_KEY`.
3. Make sure the kernel can access CUDA and the local PyTorch + Ultralytics install.
4. Keep the dataset under `notebooks/test-setup-02-train-hparams/datasets/data` so Windows path lengths stay short enough for training.

## Step 1. Initialize the notebook workspace

The notebook resolves its own folder, creates notebook-local output directories, and prints the managed layout it will use for the rest of the run.

Expected result: the notebook prints the workspace root, notebook root, notebook directory, dataset directory, results directory, model directory, logs directory, and worker count.

## Step 2. Install the required packages

The notebook installs CUDA-enabled PyTorch, Ultralytics, Roboflow, MLflow, and the support packages needed for training and dataset handling.

Expected result: package installation output followed by Ultralytics environment checks.

## Step 3. Configure MLflow logging

The notebook sets up a file-based MLflow tracking directory inside the notebook folder, creates the experiment, and defines a helper for logging parameters, metrics, and artifacts.

Expected result: MLflow prints the tracking URI and experiment name, and the tracking directory is created if needed.

## Step 4. Download and register the dataset

The notebook creates a Roboflow client from `notebooks/.env`, downloads the dataset export into the managed dataset directory, and records the dataset YAML path.

Expected result: the notebook prints the dataset root, the `data.yaml` location, and confirms the dataset exists.

## Step 5. Build the fast validation preview

The notebook removes stale Ultralytics cache files, then builds a small validation preview YAML when preview images are available. If preview images are not available, it falls back to the source dataset YAML.

Expected result: stale `labels.cache` files are removed, the fast-validation YAML is written under the notebook `_v/` folder when possible, and the notebook prints the path it will use.

## Step 6. Run preflight checks and load the model

The notebook checks CUDA visibility, confirms the model weights exist, confirms the dataset YAML exists, and loads the YOLO model for training.

Expected result: the notebook prints CUDA device information, verifies the weights and dataset config, and prepares the model for training.

## Step 7. Inspect runtime diagnostics

The notebook reports CUDA status, GPU memory, the Ultralytics version, dataset counts, and the current notebook-local paths.

Expected result: diagnostics output confirming the environment is ready.

## Step 8. Prepare the smoke test profile

The notebook builds the smoke-test training arguments and prepares the smoke model.

Expected result: the notebook prints the smoke training arguments, including the short-lived training settings.

## Step 9. Review the smoke profile configuration

The notebook prints the smoke model configuration and confirms the selected weights.

Expected result: the notebook shows the active weights path and model override information.

## Step 10. Prepare the scale-up profile

The notebook builds the larger scale-up training arguments and prepares the scale-up model.

Expected result: the notebook prints the scale-up arguments, including the notebook-local dataset YAML and results path.

## Step 11. Review the scale-up profile configuration

The notebook prints the model override details for the scale-up run.

Expected result: the notebook shows the configured YOLO model path and confirms the override is present.

## Step 12. Run scale-up training

The notebook starts the scale-up training job, measures elapsed time, and stores the results. This is the main training step for the notebook.

Expected result: Ultralytics starts training, rebuilds dataset caches if needed, trains for the configured number of epochs, and prints the training results summary. If MLflow is enabled, the run is also tracked in the notebook-local MLflow store.

## Outputs You Should Expect

After a successful run, you should see:

- A notebook-local dataset under `datasets/data`
- Training outputs under `results/yolo26/basketball-player-detection-3-14`
- Model artifacts under `models/yolo26/basketball-player-detection-3-14`
- Logs under `logs/yolo26/basketball-player-detection-3-14`
- MLflow data under `runs/mlflow`
- A rebuilt fast-validation YAML under `_v/data.fast.yaml`

## Troubleshooting

- If Roboflow download fails, check `notebooks/.env` for `ROBOFLOW_API_KEY`.
- If training fails with missing images or `labels.cache` errors, delete the cache files under `datasets/data` and rerun the preview/preflight step.
- If MLflow fails, verify the notebook is using the file-based tracking URI printed by the MLflow setup step.
- If Windows path errors return, keep the dataset root short and avoid moving the notebook into a deeper directory tree.
