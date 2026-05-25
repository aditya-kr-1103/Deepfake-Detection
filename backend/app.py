from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import (
    ALLOWED_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    OUTPUT_DIR,
    UPLOAD_DIR,
    ensure_runtime_dirs,
)
from backend.models.image_detector import DeepfakeImageDetector, DetectorRuntimeError
from backend.models.video_detector import DeepfakeVideoDetector
from backend.schemas import DetectionResponse, HealthResponse

app = FastAPI(
    title="Beyond the Pixel API",
    description="Face-centered image and video deepfake detection service.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_runtime_dirs()

image_detector = DeepfakeImageDetector()
video_detector = DeepfakeVideoDetector(image_detector=image_detector)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_ready=image_detector.is_ready())


@app.post("/detect-image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)) -> DetectionResponse:
    validate_upload(file, allowed_prefixes=("image/",))
    saved_path = save_upload(file, UPLOAD_DIR)

    try:
        return image_detector.predict(saved_path)
    except DetectorRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Image detection failed.") from exc
    finally:
        saved_path.unlink(missing_ok=True)


@app.post("/detect-video", response_model=DetectionResponse)
async def detect_video(file: UploadFile = File(...)) -> DetectionResponse:
    validate_upload(file, allowed_prefixes=("video/",))
    saved_path = save_upload(file, UPLOAD_DIR)

    try:
        return video_detector.predict(saved_path, OUTPUT_DIR)
    except DetectorRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Video detection failed.") from exc
    finally:
        saved_path.unlink(missing_ok=True)


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def validate_upload(file: UploadFile, allowed_prefixes: tuple[str, ...]) -> None:
    content_type = file.content_type or ""
    if not any(content_type.startswith(prefix) for prefix in allowed_prefixes):
        allowed = ", ".join(prefix.rstrip("/") for prefix in allowed_prefixes)
        raise HTTPException(status_code=415, detail=f"Expected {allowed} upload.")


def save_upload(file: UploadFile, upload_dir: Path) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    safe_name = f"{uuid4().hex}{suffix}"
    destination = upload_dir / safe_name
    bytes_written = 0
    with destination.open("wb") as output:
        while chunk := file.file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_SIZE_BYTES:
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload is larger than {MAX_UPLOAD_SIZE_MB} MB.",
                )
            output.write(chunk)
    return destination
