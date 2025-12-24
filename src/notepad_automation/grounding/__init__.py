"""
Vision Grounding Module

This module provides CLIP-based semantic grounding for detecting desktop icons
without relying on template matching or fixed coordinates.
"""

from .screenshot import capture_desktop_screenshot
from .detector import IconDetector, DetectionResult

__all__ = ["capture_desktop_screenshot", "IconDetector", "DetectionResult"]
