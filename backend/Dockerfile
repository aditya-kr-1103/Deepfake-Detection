FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEEPFAKE_UPLOAD_DIR=/tmp/deepfake/uploads \
    DEEPFAKE_OUTPUT_DIR=/tmp/deepfake/outputs \
    DEEPFAKE_WEIGHTS_DIR=/app/backend/weights

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements*.txt ./backend/
RUN python -m pip install --upgrade pip \
    && python -m pip install -r backend/requirements-deploy.txt

COPY backend ./backend
RUN chmod +x /app/backend/start.sh

EXPOSE 8000

CMD ["/app/backend/start.sh"]
