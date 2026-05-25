from __future__ import annotations

from backend.config import LIKELY_FAKE_THRESHOLD, LIKELY_REAL_THRESHOLD


def label_from_score(score: float) -> tuple[str, float]:
    if score < LIKELY_REAL_THRESHOLD:
        return "LIKELY REAL", 1.0 - score
    if score >= LIKELY_FAKE_THRESHOLD:
        return "LIKELY FAKE", score
    distance_from_center = abs(score - 0.5)
    confidence = 0.5 + distance_from_center
    return "SUSPICIOUS", confidence
