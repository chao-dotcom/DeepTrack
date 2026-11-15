"""Cache detector outputs once so tracker ablations avoid repeated inference."""
import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import yaml


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _validate_detections(detections: dict, frame_count: int) -> None:
    expected = {str(frame_id) for frame_id in range(1, frame_count + 1)}
    actual = set(detections)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise RuntimeError(
            f"Cache frame range invalid: missing={len(missing)}, extra={len(extra)}"
        )
    for frame_id, frame_detections in detections.items():
        if not isinstance(frame_detections, list):
            raise RuntimeError(f"Cache frame {frame_id} is not a detection list")
        for detection in frame_detections:
            if len(detection) != 5 or not all(
                isinstance(value, (int, float)) for value in detection
            ):
                raise RuntimeError(f"Invalid detection in cached frame {frame_id}")


def cache_sequence(
    model,
    sequence: Path,
    config: dict,
    output: Path,
    max_frames=None,
    checkpoint_interval: int = 10,
) -> dict:
    image_paths = sorted((sequence / "img1").glob("*.jpg"))
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise RuntimeError(f"No images found in {sequence / 'img1'}")

    detections = {}
    if output.exists():
        try:
            detections = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Cannot resume invalid cache {output}") from error
        _validate_detections(
            detections,
            min(len(detections), len(image_paths)),
        )
    first_frame = len(detections) + 1
    if first_frame > len(image_paths):
        _validate_detections(detections, len(image_paths))
    for frame_id, image_path in enumerate(image_paths[first_frame - 1:], first_frame):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Unable to read {image_path}")
        results = model(
            frame,
            conf=config["conf_threshold"],
            iou=config["iou_threshold"],
            imgsz=config["imgsz"],
            max_det=config["max_det"],
            classes=config["classes"],
            verbose=False,
        )
        detections[str(frame_id)] = [
            [
                *box.xyxy[0].cpu().numpy().tolist(),
                float(box.conf),
            ]
            for result in results
            for box in result.boxes
        ]
        if frame_id % checkpoint_interval == 0 or frame_id == len(image_paths):
            _write_json_atomic(output, detections)

    _validate_detections(detections, len(image_paths))
    non_empty = sum(bool(frame) for frame in detections.values())
    empty_rate = 1 - non_empty / len(image_paths)
    if empty_rate > 0.95:
        raise RuntimeError(
            f"{sequence.name}: suspicious empty-frame rate {empty_rate:.1%}; "
            "refusing to accept cache"
        )
    return {
        "frames": len(image_paths),
        "non_empty_frames": non_empty,
        "empty_frame_rate": empty_rate,
        "detections": sum(len(frame) for frame in detections.values()),
        "max_detections_per_frame": max(map(len, detections.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation_smoke.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--output-dir", default="outputs/detection_cache")
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    args = parser.parse_args()
    if args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be at least 1")
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    from ultralytics import YOLO

    model = YOLO(args.model)
    root = Path(config["dataset"]["root"])
    variants = config["variants"]
    variant_name = args.variant or next(iter(variants))
    if variant_name not in variants:
        raise ValueError(
            f"Unknown variant {variant_name!r}; choose from {', '.join(variants)}"
        )
    detector_config = variants[variant_name]["detection"]
    for name in args.sequence:
        output = Path(args.output_dir) / f"{name}.json"
        stats = cache_sequence(
            model,
            root / name,
            detector_config,
            output,
            args.max_frames,
            args.checkpoint_interval,
        )
        metadata = {
            "sequence": name,
            "variant": variant_name,
            "model": args.model,
            "detection": detector_config,
            "frames": stats["frames"],
            "non_empty_frames": stats["non_empty_frames"],
            "empty_frame_rate": stats["empty_frame_rate"],
            "detections": stats["detections"],
            "max_detections_per_frame": stats["max_detections_per_frame"],
            "versions": {
                "python": __import__("platform").python_version(),
                "torch": __import__("torch").__version__,
                "ultralytics": __import__("ultralytics").__version__,
                "opencv": cv2.__version__,
            },
        }
        _write_json_atomic(
            Path(args.output_dir) / f"{name}.metadata.json", metadata
        )
        print(f"cached {name} ({variant_name})")


if __name__ == "__main__":
    main()
