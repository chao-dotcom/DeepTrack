"""Evaluate the DeepSORT tracker on MOTChallenge public detections."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_mot import load_gt
from src.evaluation.mot_metrics import MOTEvaluator
from src.models.tracking.deepsort import DeepSORT


def load_detections(
    path: Path, detection_confidence: float = 1.0
) -> dict[int, list[list[float]]]:
    detections: dict[int, list[list[float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split(",")
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_number}: expected MOT CSV columns")
            try:
                frame = int(parts[0])
                x, y, width, height = map(float, parts[2:6])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid detection") from error
            detections.setdefault(frame, []).append(
                [x, y, x + width, y + height, detection_confidence]
            )
    return detections


def tracker_from_config(variant: dict) -> DeepSORT:
    tracking = variant["tracking"]
    detection = variant["detection"]
    return DeepSORT(
        max_dist=tracking["max_dist"],
        max_iou_distance=tracking["max_iou_distance"],
        max_age=tracking["max_age"],
        n_init=tracking["n_init"],
        high_confidence=tracking.get("high_confidence", 0.5),
        min_confidence=tracking.get("min_confidence", detection["conf_threshold"]),
        iou_match_threshold=tracking.get("iou_match_threshold", 0.25),
        gating_threshold=tracking.get("gating_threshold", 9.4877),
        appearance_weight=tracking.get("appearance_weight", 0.8),
        expanded_iou_scale=tracking.get("expanded_iou_scale", 1.0),
        association_mode=tracking.get("association_mode", "two_stage"),
    )


def run_sequence(
    sequence: Path,
    variant: dict,
    max_frames: int | None,
    detection_confidence: float,
) -> dict:
    public = load_detections(sequence / "det" / "det.txt", detection_confidence)
    image_paths = sorted((sequence / "img1").glob("*.jpg"))
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise RuntimeError(f"No images found in {sequence / 'img1'}")
    missing = [frame for frame in range(1, len(image_paths) + 1) if frame not in public]
    if missing:
        raise RuntimeError(
            f"{sequence.name}: public detections missing {len(missing)} image frames"
        )

    tracker = tracker_from_config(variant)
    gt = load_gt(sequence)
    evaluator = MOTEvaluator()
    gt_frames = {}
    pred_frames = {}
    start = time.perf_counter()
    for frame_id, image_path in enumerate(image_paths, 1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Unable to read {image_path}")
        tracks = tracker.update(public[frame_id], frame)
        gt_frames[frame_id] = [[*item[:4], item[4]] for item in gt.get(frame_id, [])]
        pred_frames[frame_id] = [[*item[:4], int(item[4])] for item in tracks]
        evaluator.update(
            gt_frames[frame_id],
            pred_frames[frame_id],
            frame_id,
        )

    metrics = evaluator.compute_metrics().iloc[0]
    values = {}
    for key in ("mota", "idf1", "precision", "recall", "num_fragmentations",
                "mostly_tracked", "mostly_lost"):
        values[key] = (
            float(metrics[key]) if np.isfinite(metrics[key]) else None
        )
    values.update({
        "false_positives": int(metrics["num_false_positives"]),
        "misses": int(metrics["num_misses"]),
        "id_switches": int(metrics["num_switches"]),
    })
    values.update(evaluator.compute_hota(gt_frames, pred_frames) or {})
    elapsed = time.perf_counter() - start
    return {
        "sequence": sequence.name,
        "frames": len(image_paths),
        "public_detection_count": sum(
            len(public[frame_id]) for frame_id in range(1, len(image_paths) + 1)
        ),
        "detection_confidence": detection_confidence,
        "tracker_fps": len(image_paths) / elapsed,
        "metrics": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--variant", default="updated_yolov8x_1280")
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=1.0,
        help="Synthetic confidence assigned to every public detection row",
    )
    parser.add_argument("--output-dir", default="outputs/public_detection_measurements")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    variant = config["variants"][args.variant]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    root = Path(config["dataset"]["root"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.sequence:
        result = run_sequence(
            root / name,
            variant,
            args.max_frames,
            args.detection_confidence,
        )
        result["variant"] = args.variant
        result["git_commit"] = commit
        result["worktree_dirty"] = dirty
        output = output_dir / f"{name}_{args.variant}.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(name, result["metrics"], result["tracker_fps"])


if __name__ == "__main__":
    main()
