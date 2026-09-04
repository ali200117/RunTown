"""Download the MediaPipe pose landmarker model.

Run:  uv run python scripts/download_model.py

The .task file is a multi-megabyte binary and is gitignored: it is a build
input, not source. Anyone cloning this repo runs this script once.

Source: Google's official MediaPipe model garden.
"""

import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"

# lite / full / heavy trade accuracy against inference time. We start at lite:
# squats and side steps are large, coarse movements that do not need
# fingertip precision, and we have a ~33ms budget per frame at 30 FPS.
MODELS = {
    "lite": f"{BASE_URL}/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": f"{BASE_URL}/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": f"{BASE_URL}/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def download(variant: str) -> None:
    url = MODELS[variant]
    destination = MODELS_DIR / f"pose_landmarker_{variant}.task"

    if destination.exists():
        size_mb = destination.stat().st_size / 1_000_000
        print(f"Already present: {destination.name} ({size_mb:.1f} MB)")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {variant} from {url}")
    urllib.request.urlretrieve(url, destination)

    size_mb = destination.stat().st_size / 1_000_000
    print(f"Saved {destination} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download("lite")
