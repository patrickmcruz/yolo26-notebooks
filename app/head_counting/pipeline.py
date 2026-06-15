from __future__ import annotations
import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any
import numpy as np
import statistics

from .config import PipelineConfig
from .model import YOLOModelHandler
from .video import VideoReader, VideoWriterWrapper

logger = logging.getLogger(__name__)


def format_seconds(seconds: float) -> str:
    """Formats a duration in seconds into HH:MM:SS format."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class CountingPipeline:
    """Orchestrates the environment setup, model execution, video multithreading, and output report generation."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model_handler = YOLOModelHandler(config)
        self.counts: list[int] = []
        self.video_meta: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}

    def _apply_environment(self) -> None:
        """Applies configured environment variables, ensuring directory paths exist."""
        env_vars = {
            "YOLO_CONFIG_DIR": self.config.environment.yolo_config_dir,
            "MPLCONFIGDIR": self.config.environment.mpl_config_dir,
            "TORCH_HOME": self.config.environment.torch_home,
        }

        for var_name, path_str in env_vars.items():
            if path_str:
                p = Path(path_str)
                # Resolve relative to config directory if not absolute
                if not p.is_absolute():
                    p = self.config.config_dir / p
                p.mkdir(parents=True, exist_ok=True)
                os.environ[var_name] = str(p.resolve())
                logger.info(f"Environment variable set: {var_name}={p.resolve()}")

        # Set seeding
        seed = self.config.app.seed
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed initialized to: {seed}")

    def _build_summary(self, elapsed_sec: float, processed_frames: int) -> dict[str, Any]:
        """Compiles population statistics and execution metrics into a dictionary summary."""
        if self.counts:
            p95 = float(np.percentile(np.asarray(self.counts), 95))
            summary_counts = {
                "min_people_in_frame": int(min(self.counts)),
                "max_people_in_frame": int(max(self.counts)),
                "mean_people_per_frame": round(float(statistics.fmean(self.counts)), 3),
                "median_people_per_frame": round(float(statistics.median(self.counts)), 3),
                "p95_people_per_frame": round(p95, 3),
            }
        else:
            summary_counts = {
                "min_people_in_frame": 0,
                "max_people_in_frame": 0,
                "mean_people_per_frame": 0.0,
                "median_people_per_frame": 0.0,
                "p95_people_per_frame": 0.0,
            }

        return {
            "app": self.config.app.name,
            "video": self.config.paths.video,
            "weights": self.model_handler.weights_ref,
            "device": self.config.runtime.device,
            "video_meta": self.video_meta,
            "processed_frames": processed_frames,
            "elapsed_sec": round(float(elapsed_sec), 3),
            "fps_processed": round(processed_frames / elapsed_sec, 3) if elapsed_sec > 0 else 0.0,
            "inference": self.model_handler.predict_args,
            "counts": summary_counts,
            "outputs": {
                "annotated_video": self.config.paths.annotated_video,
                "frame_counts_csv": self.config.paths.frame_counts_csv,
                "summary_json": self.config.paths.summary_json,
                "snapshots_dir": self.config.paths.snapshots_dir,
            },
        }

    def run(self) -> dict[str, Any]:
        """Main execution flow coordinating readers, batch processors, and writers."""
        # 1. Apply environment configuration and seed
        self._apply_environment()

        # 2. Setup system runtime variables & optimization flags
        self.model_handler.setup_runtime()

        # 3. Load YOLO model into CUDA device
        self.model_handler.load_model()

        # Ensure output directories exist
        out_dir = Path(self.config.paths.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 4. Initialize multithreaded readers
        vid_stride = self.config.inference.vid_stride
        batch_size = self.config.runtime.batch_size
        progress_every = self.config.output.progress_every_n_frames

        logger.info(f"Opening video file: {self.config.paths.video}")
        with VideoReader(self.config.paths.video, stride=vid_stride) as reader:
            self.video_meta = reader.metadata
            logger.info(f"Video metadata extracted: {json.dumps(self.video_meta)}")

            expected_processed = (self.video_meta["frame_count"] + vid_stride - 1) // vid_stride
            next_progress = progress_every

            # 5. Initialize multithreaded writers
            with VideoWriterWrapper(self.config, self.video_meta) as writer:
                # 6. Setup CSV writer if enabled
                csv_file = None
                csv_writer = None
                if self.config.output.save_frame_counts:
                    csv_path = Path(self.config.paths.frame_counts_csv)
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    csv_file = csv_path.open("w", newline="", encoding="utf-8")
                    csv_writer = csv.DictWriter(
                        csv_file, fieldnames=["frame_index", "timestamp_sec", "people_count"]
                    )
                    csv_writer.writeheader()

                start_time = time.time()
                batch_frames = []
                batch_indices = []
                processed_frames = 0

                try:
                    for frame, frame_idx in reader.iter_frames():
                        batch_frames.append(frame)
                        batch_indices.append(frame_idx)

                        if len(batch_frames) >= batch_size:
                            processed_frames = self._process_and_write_batch(
                                batch_frames,
                                batch_indices,
                                writer,
                                csv_writer,
                                processed_frames,
                                expected_processed,
                                start_time,
                                next_progress,
                                progress_every,
                            )
                            next_progress = ((processed_frames // progress_every) + 1) * progress_every

                    # Process remaining frames
                    if batch_frames:
                        self._process_and_write_batch(
                            batch_frames,
                            batch_indices,
                            writer,
                            csv_writer,
                            processed_frames,
                            expected_processed,
                            start_time,
                            next_progress,
                            progress_every,
                        )

                finally:
                    if csv_file is not None:
                        csv_file.close()

                elapsed_sec = time.time() - start_time
                self.summary = self._build_summary(elapsed_sec, len(self.counts))

                # 7. Write stats summary JSON report
                if self.config.output.save_summary:
                    summary_path = Path(self.config.paths.summary_json)
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    with summary_path.open("w", encoding="utf-8") as f:
                        json.dump(self.summary, f, indent=2)

                print("=" * 60)
                print("PROCESSAMENTO FINALIZADO!")
                print(f"Média final: {self.summary['fps_processed']} FPS")
                print("=" * 60)

                return self.summary

    def _process_and_write_batch(
        self,
        batch_frames: list[np.ndarray],
        batch_indices: list[int],
        writer: VideoWriterWrapper,
        csv_writer: Any | None,
        processed_frames: int,
        expected_processed: int,
        start_time: float,
        next_progress: int,
        progress_every: int,
    ) -> int:
        """Runs batch inference, collects statistics, and enqueues frames to writing threads."""
        frames = list(batch_frames)
        indices = list(batch_indices)
        batch_frames.clear()
        batch_indices.clear()

        # Fast GPU prediction
        results = self.model_handler.predict_batch(frames)

        fps = self.video_meta.get("fps", 30.0)

        for result, frame_idx in zip(results, indices):
            timestamp_sec = frame_idx / fps if fps > 0 else 0.0
            
            # Extract count based on boxes (head counting specializes in boxes)
            count = len(result.boxes) if result.boxes is not None else 0
            self.counts.append(count)

            # Write row to CSV
            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "frame_index": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "people_count": count,
                    }
                )

            # Enqueue task for background annotation and writing
            writer.write_result(result, count, frame_idx, timestamp_sec)
            processed_frames += 1

        # Periodic status logging
        if processed_frames >= next_progress:
            elapsed = time.time() - start_time
            rate = processed_frames / elapsed if elapsed > 0 else 0.0
            remaining = (expected_processed - processed_frames) / rate if rate > 0 else 0.0
            print(
                f"Processados {processed_frames}/{expected_processed} frames "
                f"({rate:0.2f} FPS) | ETA: {format_seconds(remaining)}"
            )

        return processed_frames


def run_pipeline(config_path: Path | str) -> dict[str, Any]:
    """Helper entrypoint to easily load config and execute the pipeline."""
    config = PipelineConfig.from_yaml(config_path)
    pipeline = CountingPipeline(config)
    return pipeline.run()
