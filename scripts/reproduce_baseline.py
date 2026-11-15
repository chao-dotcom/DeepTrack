"""Reproduce and validate the frozen public-detection baseline."""
import json
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "MOT20-01": {"mota": 0.4752893809763462, "idf1": 0.549705403033722},
    "MOT20-02": {"mota": 0.4602176526088586, "idf1": 0.45718588994419845},
    "MOT20-03": {"mota": 0.44116202998169984, "idf1": 0.4143672232277247},
    "MOT20-05": {"mota": 0.4498208384389737, "idf1": 0.3946916404462801},
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs" / "public_detection_measurements_reproduced"
    command = [
        sys.executable,
        str(root / "scripts" / "measure_public_detections.py"),
        "--config",
        str(root / "configs" / "evaluation.yaml"),
        "--variant",
        "updated_yolov8x_1280",
        "--output-dir",
        str(output_dir),
    ]
    for sequence in EXPECTED:
        command.extend(["--sequence", sequence])
    subprocess.run(command, cwd=root, check=True)

    for sequence, expected in EXPECTED.items():
        result_path = output_dir / f"{sequence}_updated_yolov8x_1280.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = result["metrics"]
        for name, value in expected.items():
            if abs(metrics[name] - value) > 0.0005:
                raise SystemExit(
                    f"{sequence} {name} changed: "
                    f"{metrics[name]:.8f} != {value:.8f}"
                )
        print(
            f"{sequence}: MOTA={metrics['mota']:.4%}, "
            f"IDF1={metrics['idf1']:.4%}, HOTA={metrics['hota']:.4%}, "
            f"DetA={metrics['deta']:.4%}, AssA={metrics['assa']:.4%}"
        )
    print(f"Validated frozen baseline in {output_dir}")


if __name__ == "__main__":
    main()
