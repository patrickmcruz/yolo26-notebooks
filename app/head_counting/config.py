"""
Configuration Module
~~~~~~~~~~~~~~~~~~~~

This module provides dataclass schemas and parsing methods to load, validate, and
resolve system paths for the head counting pipeline. Path parameters are dynamically
resolved relative to the directory containing the active YAML configuration file.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class AppConfig:
    """General application metadata configurations.

    Attributes:
        name: Name of the application setup for logs and reports.
        seed: Random seed for reproducibility in sampling and shuffling operations.
    """
    name: str = "head-counting-pipeline"
    seed: int = 42


@dataclass
class EnvConfig:
    """Local cache and configuration folders for machine learning libraries.

    Attributes:
        yolo_config_dir: Directory where YOLO keeps local parameters.
        mpl_config_dir: Cache folder for Matplotlib parameters.
        torch_home: Directory where PyTorch downloads pre-trained weights.
    """
    yolo_config_dir: str = ".yolo"
    mpl_config_dir: str = ".cache/matplotlib"
    torch_home: str = ".cache/torch"


@dataclass
class PathsConfig:
    """Input and output asset file paths. Relative paths are resolved to absolute.

    Attributes:
        video: Path to the input MP4/AVI video file.
        images: Glob pattern for input images (if applicable).
        weights: Path to the local YOLO model weights (.pt or .engine).
        weights_search_dirs: Directory paths searched in order to locate model weights.
        output_dir: Main folder where output files are created.
        annotated_video: Target path for the output annotated MP4 video.
        frame_counts_csv: Target path for the frame-by-frame counts CSV table.
        summary_json: Target path for the final executive summary JSON report.
        snapshots_dir: Directory where JPEG snapshots of selected frames are stored.
    """
    video: str = ""
    images: str = ""
    weights: str = "yolo26x.pt"
    weights_search_dirs: list[str] = field(default_factory=list)
    output_dir: str = "output"
    annotated_video: str = ""
    frame_counts_csv: str = ""
    summary_json: str = ""
    snapshots_dir: str = ""


@dataclass
class RuntimeConfig:
    """PyTorch device routing, multi-threading, and VRAM options.

    Attributes:
        require_cuda: If True, halts execution if CUDA is unavailable.
        device: CUDA device ID (e.g. 0 for GPU 0).
        batch_size: Number of frames processed in a single YOLO batch.
        auto_reduce_batch_on_oom: Halves batch size on CUDA Out-of-Memory.
        empty_cuda_cache_every_batches: Cache cleanup interval (0 to disable).
        torch_float32_matmul_precision: Precision level for PyTorch matrix operations.
        allow_tf32: Enables TensorFloat32 math kernels on Ampere+ GPUs (RTX 4090).
    """
    require_cuda: bool = True
    device: int = 0
    batch_size: int = 16
    auto_reduce_batch_on_oom: bool = True
    empty_cuda_cache_every_batches: int = 0
    torch_float32_matmul_precision: str = "highest"
    allow_tf32: bool = True


@dataclass
class InferenceConfig:
    """YOLO model task and confidence thresholds for object detection.

    Attributes:
        task: YOLO pipeline task (typically 'detect' for bounding boxes).
        imgsz: Resolution of frames fed into the network (e.g. 1920).
        conf: Confidence threshold limit; objects below this are discarded.
        iou: Intersection-Over-Union threshold for Non-Maximum Suppression (NMS).
        max_det: Maximum detections kept per frame.
        classes: List of integer class IDs to detect (0 corresponds to person/head).
        augment: Enables Test-Time Augmentation (TTA).
        half: Runs network inference in FP16 precision.
        verbose: Prints batch prediction details.
        vid_stride: Evaluates every N-th frame of the video.
    """
    task: str = "detect"
    imgsz: int = 1920
    conf: float = 0.18
    iou: float = 0.65
    max_det: int = 2000
    classes: list[int] = field(default_factory=lambda: [0])
    augment: bool = False
    half: bool = False
    verbose: bool = False
    vid_stride: int = 1


@dataclass
class CountingConfig:
    """Target output detection features for people counts.

    Attributes:
        count_source: Source for counts ('boxes' or 'keypoints').
        require_keypoints: Requires pose checkpoints for counts validation.
    """
    count_source: str = "boxes"
    require_keypoints: bool = False


@dataclass
class OverlayConfig:
    """Visual style settings for annotations drawn on the output video.

    Attributes:
        enabled: If True, draws annotations on frames.
        font_scale: Font scale multiplier for rendered text.
        thickness: Bounding box line thickness and text weight.
    """
    enabled: bool = True
    font_scale: float = 0.6
    thickness: int = 1


@dataclass
class OutputConfig:
    """Report and export specifications.

    Attributes:
        output_resolution: Resizes video frames to [width, height] before compression.
        save_annotated_video: Enables saving of the output MP4 video.
        save_frame_counts: Enables saving of the CSV frame records.
        save_summary: Enables saving of the JSON report summary.
        save_snapshot_every_n_frames: Interval to save JPEG audit snapshots (0 to disable).
        video_codec: OpenCV fourcc codec string (e.g. 'mp4v').
        progress_every_n_frames: Console logging interval.
        overlay: Aesthetic overlay properties.
    """
    output_resolution: list[int] | None = None
    save_annotated_video: bool = True
    save_frame_counts: bool = True
    save_summary: bool = True
    save_snapshot_every_n_frames: int = 30
    video_codec: str = "mp4v"
    progress_every_n_frames: int = 300
    overlay: OverlayConfig = field(default_factory=OverlayConfig)


@dataclass
class PipelineConfig:
    """Aggregates all system configurations into a single type-safe interface."""
    app: AppConfig = field(default_factory=AppConfig)
    environment: EnvConfig = field(default_factory=EnvConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Config folder directory (used to resolve relative paths)
    config_dir: Path = field(default_factory=lambda: Path.cwd())

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> PipelineConfig:
        """Loads configuration from a YAML file, parses and checks fields, and resolves paths.

        Args:
            yaml_path: Absolute or relative path to the target YAML file.

        Returns:
            A validated PipelineConfig instance with absolute path settings.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        yaml_path = Path(yaml_path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        config_dir = yaml_path.parent

        # App Configuration
        raw_app = raw.get("app", {})
        app = AppConfig(
            name=raw_app.get("name", "head-counting-pipeline"),
            seed=int(raw_app.get("seed", 42)),
        )

        # Environment Configuration
        raw_env = raw.get("environment", {})
        environment = EnvConfig(
            yolo_config_dir=raw_env.get("YOLO_CONFIG_DIR", ".yolo"),
            mpl_config_dir=raw_env.get("MPLCONFIGDIR", ".cache/matplotlib"),
            torch_home=raw_env.get("TORCH_HOME", ".cache/torch"),
        )

        # Paths Configuration
        raw_paths = raw.get("paths", {})
        paths = PathsConfig(
            video=raw_paths.get("video", ""),
            images=raw_paths.get("images", ""),
            weights=raw_paths.get("weights", "yolo26x.pt"),
            weights_search_dirs=list(raw_paths.get("weights_search_dirs", [])),
            output_dir=raw_paths.get("output_dir", "output"),
            annotated_video=raw_paths.get("annotated_video", ""),
            frame_counts_csv=raw_paths.get("frame_counts_csv", ""),
            summary_json=raw_paths.get("summary_json", ""),
            snapshots_dir=raw_paths.get("snapshots_dir", ""),
        )

        # Runtime Configuration
        raw_runtime = raw.get("runtime", {})
        runtime = RuntimeConfig(
            require_cuda=bool(raw_runtime.get("require_cuda", True)),
            device=int(raw_runtime.get("device", 0)),
            batch_size=int(raw_runtime.get("batch_size", 16)),
            auto_reduce_batch_on_oom=bool(raw_runtime.get("auto_reduce_batch_on_oom", True)),
            empty_cuda_cache_every_batches=int(raw_runtime.get("empty_cuda_cache_every_batches", 0)),
            torch_float32_matmul_precision=raw_runtime.get("torch_float32_matmul_precision", "highest"),
            allow_tf32=bool(raw_runtime.get("allow_tf32", True)),
        )

        # Inference Configuration
        raw_inference = raw.get("inference", {})
        inference = InferenceConfig(
            task=raw_inference.get("task", "detect"),
            imgsz=int(raw_inference.get("imgsz", 1920)),
            conf=float(raw_inference.get("conf", 0.18)),
            iou=float(raw_inference.get("iou", 0.65)),
            max_det=int(raw_inference.get("max_det", 2000)),
            classes=list(raw_inference.get("classes", [0])),
            augment=bool(raw_inference.get("augment", False)),
            half=bool(raw_inference.get("half", False)),
            verbose=bool(raw_inference.get("verbose", False)),
            vid_stride=int(raw_inference.get("vid_stride", 1)),
        )

        # Counting Configuration
        raw_counting = raw.get("counting", {})
        counting = CountingConfig(
            count_source=raw_counting.get("count_source", "boxes"),
            require_keypoints=bool(raw_counting.get("require_keypoints", False)),
        )

        # Output & Overlay Configuration
        raw_output = raw.get("output", {})
        raw_overlay = raw_output.get("overlay", {})
        overlay = OverlayConfig(
            enabled=bool(raw_overlay.get("enabled", True)),
            font_scale=float(raw_overlay.get("font_scale", 0.6)),
            thickness=int(raw_overlay.get("thickness", 1)),
        )
        
        output = OutputConfig(
            output_resolution=raw_output.get("output_resolution"),
            save_annotated_video=bool(raw_output.get("save_annotated_video", True)),
            save_frame_counts=bool(raw_output.get("save_frame_counts", True)),
            save_summary=bool(raw_output.get("save_summary", True)),
            save_snapshot_every_n_frames=int(raw_output.get("save_snapshot_every_n_frames", 30)),
            video_codec=raw_output.get("video_codec", "mp4v"),
            progress_every_n_frames=int(raw_output.get("progress_every_n_frames", 300)),
            overlay=overlay,
        )

        config = cls(
            app=app,
            environment=environment,
            paths=paths,
            runtime=runtime,
            inference=inference,
            counting=counting,
            output=output,
            config_dir=config_dir,
        )
        config._resolve_paths()
        return config

    def _resolve_paths(self) -> None:
        """Resolves raw relative paths in settings into absolute paths relative to the config file's parent folder."""
        
        def to_absolute(val: str) -> str:
            if not val:
                return ""
            p = Path(val)
            if p.is_absolute():
                return str(p.resolve())
            return str((self.config_dir / p).resolve())

        # Resolve paths in PathConfig
        self.paths.video = to_absolute(self.paths.video)
        self.paths.images = to_absolute(self.paths.images)
        self.paths.weights = to_absolute(self.paths.weights)
        self.paths.weights_search_dirs = [to_absolute(d) for d in self.paths.weights_search_dirs]
        self.paths.output_dir = to_absolute(self.paths.output_dir)
        
        # Setup defaults for derived paths if not custom-specified
        out_dir = Path(self.paths.output_dir)
        
        if not self.paths.annotated_video:
            video_name = Path(self.paths.video).stem if self.paths.video else "video"
            self.paths.annotated_video = str(out_dir / f"{video_name}_annotated.mp4")
        else:
            self.paths.annotated_video = to_absolute(self.paths.annotated_video)
            
        if not self.paths.frame_counts_csv:
            self.paths.frame_counts_csv = str(out_dir / "frame_counts.csv")
        else:
            self.paths.frame_counts_csv = to_absolute(self.paths.frame_counts_csv)
            
        if not self.paths.summary_json:
            self.paths.summary_json = str(out_dir / "summary.json")
        else:
            self.paths.summary_json = to_absolute(self.paths.summary_json)
            
        if not self.paths.snapshots_dir:
            self.paths.snapshots_dir = str(out_dir / "snapshots")
        else:
            self.paths.snapshots_dir = to_absolute(self.paths.snapshots_dir)
