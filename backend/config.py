from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

UPLOAD_DIR = Path(os.getenv("DEEPFAKE_UPLOAD_DIR", str(BASE_DIR / "uploads")))
OUTPUT_DIR = Path(os.getenv("DEEPFAKE_OUTPUT_DIR", str(BASE_DIR / "outputs")))
WEIGHTS_DIR = Path(os.getenv("DEEPFAKE_WEIGHTS_DIR", str(BASE_DIR / "weights")))

MODEL_PATH = Path(
    os.getenv("DEEPFAKE_MODEL_PATH", str(WEIGHTS_DIR / "dfdc_effnet_b7.pt"))
)

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("DEEPFAKE_CORS_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS)).split(",")
    if origin.strip()
]
CORS_ALLOW_CREDENTIALS = os.getenv("DEEPFAKE_CORS_ALLOW_CREDENTIALS", "false").lower() in {
    "1",
    "true",
    "yes",
}

IMAGE_SIZE = 380
VIDEO_FRAME_STRIDE = 5
LIKELY_REAL_THRESHOLD = 0.40
LIKELY_FAKE_THRESHOLD = 0.60
MAX_UPLOAD_SIZE_MB = int(os.getenv("DEEPFAKE_MAX_UPLOAD_SIZE_MB", "250"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
