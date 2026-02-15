"""
Retargeting Wuji Module

This module provides retargeting utilities for Wuji hand, including:
- DexPilot retargeting with MediaPipe format conversion
"""

from .mediapipe import apply_mediapipe_transformations

# `WujiHandRetargeter` depends on Pinocchio. Keep package import lightweight so that
# utilities like MediaPipe conversion can be used without requiring Pinocchio.
try:
    from .retarget import WujiHandRetargeter, RetargetingResult
except Exception:  # pragma: no cover - best-effort optional dependency
    WujiHandRetargeter = None  # type: ignore
    RetargetingResult = None  # type: ignore

__all__ = [
    'WujiHandRetargeter',
    'RetargetingResult',
    'apply_mediapipe_transformations',
]
