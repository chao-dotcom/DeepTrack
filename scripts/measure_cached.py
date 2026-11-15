"""Run tracker measurements against cached detections."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.measure_mot import load_gt, match_count
from src.evaluation.mot_metrics import MOTEvaluator
from src.models.tracking.deepsort import DeepSORT


def run(
    sequence: Path,
    cache: Path,
    variant: dict,
    iou_threshold: float,
    max_frames: int | None = None,
) -> dict:
    tracking = variant["tracking"]
    detection = variant["detection"]
    tracker = DeepSORT(
        max_dist=tracking["max_dist"],
        max_iou_distance=tracking["max_iou_distance"],
        max_age=tracking["max_age"],
        n_init=tracking["n_init"],
        high_confidence=tracking.get("high_confidence", 0.5),
        min_confidence=tracking.get(
            "min_confidence", detection["conf_threshold"]
        ),
        iou_match_threshold=tracking.get("iou_match_threshold", 0.25),
        gating_threshold=tracking.get("gating_threshold", 9.4877),
        appearance_weight=tracking.get("appearance_weight", 0.8),
        expanded_iou_scale=tracking.get("expanded_iou_scale", 1.0),
        association_mode=tracking.get("association_mode", "two_stage"),
    )
    detections = json.loads(cache.read_text(encoding="utf-8"))
    gt = load_gt(sequence)
    evaluator = MOTEvaluator()
    funnel = {
        "gt": 0,
        "detector_after_confidence": 0,
        "matched_existing_track": 0,
        "tentative_or_confirmed_output": 0,
        "matched_confirmed_output": 0,
    }
    start = time.perf_counter()
    image_paths = sorted((sequence / "img1").glob("*.jpg"))
    image_paths = image_paths[:len(detections)]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    for frame_id, image_path in enumerate(image_paths, 1):
        frame = cv2.imread(str(image_path))
        frame_detections = detections.get(str(frame_id), [])
        tracks = tracker.update(frame_detections, frame)
        truth = gt.get(frame_id, [])
        truth_boxes = [item[:4] for item in truth]
        funnel["gt"] += len(truth_boxes)
        funnel["detector_after_confidence"] += match_count(
            truth_boxes, [item[:4] for item in frame_detections], iou_threshold
        )
        stats = tracker.last_update_stats
        funnel["matched_existing_track"] += match_count(
            truth_boxes, stats["matched_existing_tracks"], iou_threshold
        )
        funnel["tentative_or_confirmed_output"] += match_count(
            truth_boxes, stats["tentative_or_confirmed_output"], iou_threshold
        )
        funnel["matched_confirmed_output"] += match_count(
            truth_boxes, stats["confirmed_output"], iou_threshold
        )
        evaluator.update(
            [[*item[:4], item[4]] for item in truth],
            [[*item[:4], int(item[4])] for item in tracks],
            frame_id,
        )
    metrics = evaluator.compute_metrics().iloc[0]
    return {
        "frames": len(image_paths),
        "seconds": time.perf_counter() - start,
        "tracker_fps": len(image_paths) / (time.perf_counter() - start),
        "funnel": funnel,
        "metrics": {
            "mota": float(metrics["mota"]),
            "idf1": float(metrics["idf1"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "false_positives": int(metrics["num_false_positives"]),
            "misses": int(metrics["num_misses"]),
            "id_switches": int(metrics["num_switches"]),
            "fragmentations": int(metrics["num_fragmentations"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation_smoke.yaml")
    parser.add_argument("--cache-dir", default="outputs/detection_cache")
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--variant", default="smoke_yolov8n_1280")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    variant = config["variants"][args.variant]
    root = Path(config["dataset"]["root"])
    for name in args.sequence:
        result = run(
            root / name,
            Path(args.cache_dir) / f"{name}.json",
            variant,
            config["dataset"]["iou_threshold"],
            args.max_frames,
        )
        output = Path(args.cache_dir) / f"{name}_{args.variant}_metrics.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(name, result["metrics"], result["tracker_fps"])


if __name__ == "__main__":
    main()
