"""Reproducible full-sequence MOT20 measurement and recall funnel."""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.deepsort_tracker import DeepSORTVideoTracker
from src.evaluation.mot_metrics import MOTEvaluator


def load_gt(sequence: Path) -> Dict[int, List[List[float]]]:
    gt = {}
    with open(sequence / "gt" / "gt.txt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            if float(parts[6]) != 1 or int(float(parts[7])) != 1:
                continue
            x, y, width, height = map(float, parts[2:6])
            gt.setdefault(int(parts[0]), []).append(
                [x, y, x + width, y + height, int(parts[1])]
            )
    return gt


def iou(first: List[float], second: List[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_count(gt_boxes, predicted_boxes, threshold: float) -> int:
    if not gt_boxes or not predicted_boxes:
        return 0
    costs = np.array([[1 - iou(gt, pred) for pred in predicted_boxes] for gt in gt_boxes])
    rows, columns = linear_sum_assignment(costs)
    return int(sum(
        costs[row, column] <= 1 - threshold for row, column in zip(rows, columns)
    ))


def run_variant(name: str, variant: dict, dataset_root: Path, sequences: List[str],
                iou_threshold: float, output_dir: Path, max_frames=None) -> dict:
    detector_config = variant["detection"]
    config = {"detection": detector_config, "tracking": variant["tracking"]}
    tracker = DeepSORTVideoTracker(variant["model_path"], config=config)
    totals = {
        "gt": 0, "detector_after_nms": 0, "detector_after_confidence": 0,
        "matched_existing_track": 0,
        "tentative_or_confirmed_output": 0,
        "matched_confirmed_output": 0, "frames": 0, "seconds": 0.0,
        "raw_detection_count": 0, "confidence_detection_count": 0,
        "max_raw_detections": 0, "max_confidence_detections": 0,
    }
    sequence_results = {}
    start = time.perf_counter()

    for sequence_name in sequences:
        sequence_start = time.perf_counter()
        sequence = dataset_root / sequence_name
        image_paths = sorted((sequence / "img1").glob("*.jpg"))
        if max_frames is not None:
            image_paths = image_paths[:max_frames]
        gt = load_gt(sequence)
        evaluator = MOTEvaluator()
        funnel = {key: 0 for key in totals if key not in ("seconds",)}
        for frame_id, image_path in enumerate(image_paths, 1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Unable to read {image_path}")
            # Low-confidence inference provides the candidate pool. Ultralytics
            # applies NMS in this call; the confidence stage is explicit below.
            raw_results = tracker.detector(
                frame, conf=0.001, iou=detector_config["iou_threshold"],
                imgsz=detector_config["imgsz"], max_det=detector_config["max_det"],
                classes=detector_config["classes"], verbose=False,
            )
            raw_boxes = [
                box.xyxy[0].cpu().numpy().tolist()
                for result in raw_results for box in result.boxes
            ]
            detections = [
                [
                    *box.xyxy[0].cpu().numpy().tolist(),
                    float(box.conf),
                ]
                for result in raw_results
                for box in result.boxes
                if float(box.conf) >= detector_config["conf_threshold"]
            ]
            raw_detection_count = len(raw_boxes)
            confidence_detection_count = len(detections)
            totals["raw_detection_count"] += raw_detection_count
            totals["confidence_detection_count"] += confidence_detection_count
            totals["max_raw_detections"] = max(
                totals["max_raw_detections"], raw_detection_count
            )
            totals["max_confidence_detections"] = max(
                totals["max_confidence_detections"], confidence_detection_count
            )
            detection_boxes = [d[:4] for d in detections]
            tracks = tracker.tracker.update(detections, frame)
            output_boxes = [track[:4] for track in tracks]
            truth = [item[:4] for item in gt.get(frame_id, [])]
            evaluator.update(
                [[*item[:4], item[4]] for item in gt.get(frame_id, [])],
                [[*item[:4], int(item[4])] for item in tracks],
                frame_id,
            )

            funnel["gt"] += int(len(truth))
            funnel["detector_after_nms"] += match_count(truth, raw_boxes, iou_threshold)
            funnel["detector_after_confidence"] += match_count(
                truth, detection_boxes, iou_threshold
            )
            funnel["matched_existing_track"] += match_count(
                truth, tracker.tracker.last_update_stats["matched_existing_tracks"], iou_threshold
            )
            funnel["tentative_or_confirmed_output"] += match_count(
                truth,
                tracker.tracker.last_update_stats["tentative_or_confirmed_output"],
                iou_threshold,
            )
            funnel["matched_confirmed_output"] += match_count(
                truth, tracker.tracker.last_update_stats["confirmed_output"], iou_threshold
            )
            funnel["frames"] += 1
        for key in funnel:
            totals[key] += funnel[key]
        metrics = evaluator.compute_metrics()
        if hasattr(metrics, "iloc"):
            row = metrics.iloc[0]
            metric_values = {
                key: float(row[key]) if np.isfinite(row[key]) else None
                for key in (
                    "mota", "motp", "precision", "recall", "idf1",
                    "num_false_positives", "num_misses", "num_fragmentations",
                    "mostly_tracked", "mostly_lost",
                )
            }
            metric_values["id_switches"] = int(row["num_switches"])
        else:
            metric_values = {
                key: float(metrics[key]["Overall"])
                for key in ("mota", "motp", "precision", "recall", "idf1")
            }
            metric_values.update({
                "num_false_positives": int(metrics["num_false_positives"]["Overall"]),
                "num_misses": int(metrics["num_misses"]["Overall"]),
                "num_fragmentations": 0,
                "mostly_tracked": 0,
                "mostly_lost": 0,
                "id_switches": int(metrics["num_switches"]["Overall"]),
            })
        metric_values["fps"] = len(image_paths) / (time.perf_counter() - sequence_start)
        for key, value in metric_values.items():
            if value is not None and key not in sequence_results.get(sequence_name, {}):
                totals[key] = totals.get(key, 0) + value
        sequence_results[sequence_name] = {
            "funnel": funnel, "metrics": metric_values, "frames": len(image_paths)
        }

    totals["seconds"] = time.perf_counter() - start
    totals["detector_recall"] = totals["detector_after_confidence"] / totals["gt"] if totals["gt"] else 0
    totals["confirmed_recall"] = totals["matched_confirmed_output"] / totals["gt"] if totals["gt"] else 0
    totals["fps"] = totals["frames"] / totals["seconds"] if totals["seconds"] else 0
    result = {"variant": name, "totals": totals, "sequences": sequence_results}
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{name}.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--sequence", action="append",
                        help="Sequence to measure; may be supplied more than once")
    parser.add_argument("--max-frames", type=int,
                        help="Limit frames per sequence for a smoke test")
    parser.add_argument("--full-sequence", action="store_true",
                        help="Ignore the config frame limit and process every frame")
    parser.add_argument("--variant", action="append",
                        help="Variant to run; may be supplied more than once")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    wandb_run = None
    if config["run"].get("wandb"):
        try:
            import wandb
            wandb_run = wandb.init(project=config["run"]["wandb_project"],
                                   config=config)
        except ImportError as exc:
            raise RuntimeError("W&B logging requested but wandb is not installed") from exc
    root = Path(config["dataset"]["root"])
    output_dir = Path(config["run"]["output_dir"])
    sequences = args.sequence or config["dataset"]["sequences"]
    max_frames = None if args.full_sequence else (
        args.max_frames if args.max_frames is not None else config["run"]["max_frames"]
    )
    variants = config["variants"]
    if args.variant:
        variants = {
            name: variant for name, variant in variants.items()
            if name in args.variant
        }
        missing = set(args.variant) - set(variants)
        if missing:
            parser.error(f"unknown variant(s): {', '.join(sorted(missing))}")
    results = [
        run_variant(name, variant, root, sequences,
                    config["dataset"]["iou_threshold"], output_dir,
                    max_frames)
        for name, variant in variants.items()
    ]
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    if wandb_run is not None:
        for result in results:
            wandb_run.log({
                f"{result['variant']}/{key}": value
                for key, value in result["totals"].items()
                if isinstance(value, (int, float))
            })
        wandb_run.finish()
    for result in results:
        print(result["variant"], result["totals"])


if __name__ == "__main__":
    main()
