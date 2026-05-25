from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from backend.config import VIDEO_FRAME_STRIDE
from backend.models.image_detector import DeepfakeImageDetector, DetectorRuntimeError
from backend.models.labels import label_from_score
from backend.schemas import DetectionResponse


class DeepfakeVideoDetector:
    def __init__(self, image_detector: DeepfakeImageDetector) -> None:
        self.image_detector = image_detector

    def predict(self, video_path: Path, output_dir: Path) -> DetectionResponse:
        try:
            import cv2
        except ImportError as exc:
            raise DetectorRuntimeError(
                "OpenCV is not installed. Install backend requirements first."
            ) from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Could not open video file.")

        scores: list[float] = []
        frame_index = 0
        session_dir = output_dir / uuid4().hex
        session_dir.mkdir(parents=True, exist_ok=True)

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % VIDEO_FRAME_STRIDE != 0:
                    continue

                frame_path = session_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                try:
                    result = self.image_detector.predict(frame_path)
                except ValueError:
                    continue
                scores.append(result.score)
        finally:
            capture.release()
            shutil.rmtree(session_dir, ignore_errors=True)

        if not scores:
            raise ValueError("No usable face frames found in the uploaded video.")

        score = sum(scores) / len(scores)
        prediction, confidence = label_from_score(score)
        return DetectionResponse(
            prediction=prediction,
            score=score,
            confidence=confidence,
            media_type="video",
            frames_used=len(scores),
        )
