# Deployment

This project is deployable as two services:

- `backend`: FastAPI + PyTorch inference API.
- `frontend`: Vite static build served by Nginx or any static host.

## Required Model File

The checkpoint is not committed to git. Place it at:

```text
backend/weights/dfdc_effnet_b7.pt
```

For Docker deployments, either mount that directory or set `DEEPFAKE_MODEL_URL` to a private URL that the backend can download at startup.

## Docker Compose

Local production-style run:

```powershell
docker compose up --build
```

Open:

```text
http://localhost:5173
```

The backend is exposed at:

```text
http://localhost:8000
```

For a hosted VM, update `docker-compose.yml`:

- Set frontend build arg `VITE_API_BASE_URL` to the public backend URL.
- Set backend `DEEPFAKE_CORS_ORIGINS` to the public frontend URL.
- Mount `backend/weights` or set `DEEPFAKE_MODEL_URL`.

## Backend Environment

```text
DEEPFAKE_CORS_ORIGINS=https://your-frontend.example.com
DEEPFAKE_CORS_ALLOW_CREDENTIALS=false
DEEPFAKE_MODEL_PATH=/app/backend/weights/dfdc_effnet_b7.pt
DEEPFAKE_MODEL_URL=https://your-private-storage.example.com/dfdc_effnet_b7.pt
DEEPFAKE_MAX_UPLOAD_SIZE_MB=250
DEEPFAKE_UPLOAD_DIR=/tmp/deepfake/uploads
DEEPFAKE_OUTPUT_DIR=/tmp/deepfake/outputs
PORT=8000
```

Backend start command:

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port "$PORT"
```

Install dependencies for deployment:

```bash
python -m pip install -r backend/requirements-deploy.txt
```

## Frontend Environment

```text
VITE_API_BASE_URL=https://your-backend.example.com
```

Static host settings:

```text
Build command: npm ci && npm run build
Publish directory: dist
```

## Host Notes

- Use a backend host that supports long-running Python services and enough memory for PyTorch + EfficientNet-B7.
- Serverless functions are usually a poor fit for this backend because the model is large and inference is slow on CPU.
- `backend/uploads` and `backend/outputs` are temporary runtime paths. The API deletes request uploads and extracted video frames after each request.
