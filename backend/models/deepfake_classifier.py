from __future__ import annotations

from pathlib import Path
from typing import Any


class ModelLoadError(RuntimeError):
    """Raised when the classifier cannot be initialized."""


class DeepFakeClassifier:
    """EfficientNet-B7 binary classifier used by the DFDC-style pipeline.

    The paper points to Selim Seferbekov's DFDC solution. Those checkpoints use
    an `encoder + avg_pool + dropout + fc` module layout around
    `tf_efficientnet_b7_ns`, so this wrapper mirrors that layout when loading
    the public release weights.
    """

    def __init__(self, checkpoint_path: Path, device: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device_name = device
        self._model: Any | None = None
        self._torch: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def torch(self) -> Any:
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise ModelLoadError(
                    "PyTorch is not installed. Install backend requirements first."
                ) from exc
            self._torch = torch
        return self._torch

    @property
    def device(self) -> Any:
        torch = self.torch
        if self.device_name:
            return torch.device(self.device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_probability(self, face_tensor: Any) -> float:
        torch = self.torch
        model = self.model
        with torch.no_grad():
            logits = model(face_tensor.to(self.device))
            probability = torch.sigmoid(logits).flatten()[0].item()
        return float(probability)

    def _load_model(self) -> Any:
        if not self.checkpoint_path.exists():
            raise ModelLoadError(
                f"Model checkpoint not found at {self.checkpoint_path}. "
                "Place the DFDC EfficientNet-B7 weights there or set DEEPFAKE_MODEL_PATH."
            )

        torch = self.torch
        model = self._build_model()
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_dataparallel_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def _build_model(self) -> Any:
        try:
            import timm
            from torch import nn
        except ImportError as exc:
            raise ModelLoadError(
                "timm is not installed. Install backend requirements first."
            ) from exc

        class DFDCClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = timm.create_model(
                    "tf_efficientnet_b7_ns",
                    pretrained=False,
                    drop_path_rate=0.2,
                )
                self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.dropout = nn.Dropout(0.0)
                self.fc = nn.Linear(2560, 1)

            def forward(self, x: Any) -> Any:
                x = self.encoder.forward_features(x)
                x = self.avg_pool(x).flatten(1)
                x = self.dropout(x)
                return self.fc(x)

        return DFDCClassifier()

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
        if isinstance(checkpoint, dict):
            return checkpoint
        raise ModelLoadError("Unsupported checkpoint format.")

    @staticmethod
    def _strip_dataparallel_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            cleaned[key] = value
        return cleaned
