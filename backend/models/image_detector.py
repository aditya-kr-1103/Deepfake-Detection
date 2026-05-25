from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.config import IMAGE_SIZE, MODEL_PATH
from backend.models.deepfake_classifier import DeepFakeClassifier, ModelLoadError
from backend.models.labels import label_from_score
from backend.schemas import DetectionResponse


class DetectorRuntimeError(RuntimeError):
    """Raised for missing runtime dependencies or model assets."""


class DeepfakeImageDetector:
    def __init__(self, classifier: DeepFakeClassifier | None = None) -> None:
        self.classifier = classifier or DeepFakeClassifier(MODEL_PATH)
        self._mtcnn: Any | None = None

    def is_ready(self) -> bool:
        return MODEL_PATH.exists()

    def predict(self, image_path: Path) -> DetectionResponse:
        try:
            face_tensor = self._extract_face_tensor(image_path)
            score = self.classifier.predict_probability(face_tensor)
        except ModelLoadError as exc:
            raise DetectorRuntimeError(str(exc)) from exc

        prediction, confidence = label_from_score(score)
        return DetectionResponse(
            prediction=prediction,
            score=score,
            confidence=confidence,
            media_type="image",
        )

    def _extract_face_tensor(self, image_path: Path) -> Any:
        try:
            import cv2
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as exc:
            raise DetectorRuntimeError(
                "Image detection dependencies are missing. Install backend requirements first."
            ) from exc

        image = Image.open(image_path).convert("RGB")
        face_np = self._extract_face_array(image)
        if face_np is None:
            raise ValueError("No detectable face found. Use media with a clear frontal face.")

        resized = cv2.resize(face_np, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _extract_face_array(self, image: Any) -> Any | None:
        try:
            face = self.mtcnn(image)
        except DetectorRuntimeError:
            face = None

        if face is not None:
            return face.permute(1, 2, 0).mul(255).byte().cpu().numpy()

        return self._extract_face_with_opencv(image)

    def _extract_face_with_opencv(self, image: Any) -> Any | None:
        import cv2
        import numpy as np

        rgb = np.array(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None

        x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
        pad = int(max(width, height) * 0.12)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + width + pad, rgb.shape[1])
        y2 = min(y + height + pad, rgb.shape[0])
        return rgb[y1:y2, x1:x2]

    @property
    def mtcnn(self) -> Any:
        if self._mtcnn is None:
            try:
                from facenet_pytorch import MTCNN
            except ImportError as exc:
                raise DetectorRuntimeError("facenet-pytorch is not installed.") from exc
            device = self.classifier.device
            self._mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=0, post_process=False, device=device)
        return self._mtcnn
