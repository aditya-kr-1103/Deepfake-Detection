# Beyond the Pixel

Full-stack deepfake detection project based on `Research_Paper.pdf`.

## Architecture

- Backend: FastAPI service with `/detect-image`, `/detect-video`, and `/health`.
- Model path: face-centered crop, `380 x 380` preprocessing, EfficientNet-B7 binary classifier.
- Video path: samples every fifth frame, runs image detection on usable face frames, averages frame scores.
- Frontend: React + TypeScript + Vite console for image and video uploads.

## Required Model Checkpoint

The paper describes the architecture and inference pipeline, but it does not include trained weights. Put the DFDC-trained EfficientNet-B7 checkpoint at:

```text
backend/weights/dfdc_effnet_b7.pt
```

Or set an explicit path before starting the backend:

```powershell
$env:DEEPFAKE_MODEL_PATH="C:\path\to\checkpoint.pt"
```

## Backend Setup

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

API docs will be available at:

```text
http://127.0.0.1:8000/docs
```

Install the ML detector dependencies separately:

```powershell
python -m pip install -r backend\requirements-ml.txt
```

The API can start without those packages, but detection requests need them plus the checkpoint.

## Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Deployment

Deployment files are included for a split backend/frontend release:

- `backend/Dockerfile` for the FastAPI + PyTorch service.
- `frontend/Dockerfile` for the static Vite build served by Nginx.
- `docker-compose.yml` for a production-style local or VM deployment.
- `.env.example` files for required hosted URLs and runtime paths.

Read `DEPLOYMENT.md` before hosting. At minimum, set:

```text
VITE_API_BASE_URL=https://your-backend.example.com
DEEPFAKE_CORS_ORIGINS=https://your-frontend.example.com
```

The model checkpoint is not committed to git, so the backend host must receive `dfdc_effnet_b7.pt` through a mounted volume, persistent storage, or `DEEPFAKE_MODEL_URL`.

## Output Labels

- `LIKELY REAL`: score `< 0.40`
- `SUSPICIOUS`: score `0.40 - 0.60`
- `LIKELY FAKE`: score `>= 0.60`

## Notes

- CPU inference will be slow for EfficientNet-B7 and video processing.
- The detector returns an error when no face is found instead of guessing from the background.
- Python 3.14 may not be supported by every ML dependency. If installation fails for PyTorch, create the virtual environment with Python 3.11 or 3.12.
- The backend tries MTCNN if `facenet-pytorch` is installed and falls back to OpenCV Haar face detection otherwise.
