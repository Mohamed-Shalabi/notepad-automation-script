"""
LIP-based semantic grounding allows us to find UI elements
without relying on fixed coordinates or template matching.

Provides fast, reliable desktop screenshot capture using the mss library.
Handles multi-monitor setups by focusing on the primary monitor.
"""

import logging
from typing import Tuple, Optional
from dataclasses import dataclass

import mss
import numpy as np
from PIL import Image

from ..config import config


logger = logging.getLogger(__name__)


@dataclass
class ScreenshotResult:
    """Result of a screenshot capture operation."""
    image: Image.Image  # PIL Image (RGB)
    array: np.ndarray   # NumPy array (RGB, HWC format)
    width: int
    height: int
    monitor_index: int


def capture_desktop_screenshot(
    monitor_index: Optional[int] = None
) -> ScreenshotResult:
    """
    Capture a screenshot of the desktop.
    
    Args:
        monitor_index: Which monitor to capture (1-indexed). 
                      None uses the primary monitor from config.
    
    Returns:
        ScreenshotResult containing the image in multiple formats.
    
    Raises:
        RuntimeError: If screenshot capture fails.
    """
    if monitor_index is None:
        monitor_index = config.screen.primary_monitor
    
    logger.info(f"Capturing screenshot of monitor {monitor_index}")
    
    try:
        with mss.mss() as sct:
            # Get monitor info (0 is "all monitors", 1+ are individual)
            if monitor_index >= len(sct.monitors):
                logger.warning(
                    f"Monitor {monitor_index} not found, using primary (1)"
                )
                monitor_index = 1
            
            monitor = sct.monitors[monitor_index]
            logger.debug(f"Monitor specs: {monitor}")
            
            # Capture the screenshot
            screenshot = sct.grab(monitor)
            
            # Convert to PIL Image (BGRA -> RGB)
            img = Image.frombytes(
                "RGB",
                screenshot.size,
                screenshot.bgra,
                "raw",
                "BGRX"
            )
            
            # Convert to numpy array
            arr = np.array(img)
            
            result = ScreenshotResult(
                image=img,
                array=arr,
                width=screenshot.width,
                height=screenshot.height,
                monitor_index=monitor_index
            )
            
            logger.info(
                f"Screenshot captured: {result.width}x{result.height}"
            )
            
            return result
            
    except Exception as e:
        logger.error(f"Failed to capture screenshot: {e}")
        raise RuntimeError(f"Screenshot capture failed: {e}") from e


def save_screenshot(
    result: ScreenshotResult,
    filepath: str,
    annotations: Optional[list] = None
) -> None:
    """
    Save a screenshot to disk, optionally with annotations.
    
    Args:
        result: The screenshot result to save.
        filepath: Path to save the image.
        annotations: Optional list of (x, y, w, h, label, confidence) tuples
                    to draw on the image.
    """
    img = result.image.copy()
    
    if annotations:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a nicer font
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        
        for ann in annotations:
            x, y, w, h = ann[:4]
            label = ann[4] if len(ann) > 4 else ""
            conf = ann[5] if len(ann) > 5 else 0.0
            
            # Draw bounding box
            draw.rectangle(
                [x, y, x + w, y + h],
                outline="red",
                width=3
            )
            
            # Draw label with confidence
            text = f"{label}: {conf:.2f}" if label else f"{conf:.2f}"
            draw.text((x, y - 20), text, fill="red", font=font)
            
            # Draw center point
            cx, cy = x + w // 2, y + h // 2
            draw.ellipse(
                [cx - 5, cy - 5, cx + 5, cy + 5],
                fill="green",
                outline="white"
            )
    
    img.save(filepath)
    logger.info(f"Screenshot saved to: {filepath}")


def get_desktop_region(
    x: int, y: int, width: int, height: int
) -> Tuple[Image.Image, np.ndarray]:
    """
    Capture a specific region of the desktop.
    
    Args:
        x: Left coordinate.
        y: Top coordinate.
        width: Region width.
        height: Region height.
    
    Returns:
        Tuple of (PIL Image, numpy array).
    """
    logger.debug(f"Capturing region: ({x}, {y}, {width}, {height})")
    
    with mss.mss() as sct:
        region = {
            "left": x,
            "top": y,
            "width": width,
            "height": height
        }
        screenshot = sct.grab(region)
        
        img = Image.frombytes(
            "RGB",
            screenshot.size,
            screenshot.bgra,
            "raw",
            "BGRX"
        )
        
        return img, np.array(img)
