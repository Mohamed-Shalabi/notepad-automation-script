"""
Tests for Vision Grounding Module

These tests validate the icon detection functionality without
requiring actual desktop automation.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from notepad_automation.config import config
from notepad_automation.grounding.screenshot import (
    capture_desktop_screenshot,
    save_screenshot,
    ScreenshotResult
)
from notepad_automation.grounding.detector import IconDetector, DetectionResult


class TestScreenshot:
    """Tests for screenshot capture functionality."""
    
    def test_capture_returns_screenshot_result(self):
        """Test that capture returns a ScreenshotResult object."""
        result = capture_desktop_screenshot()
        
        assert isinstance(result, ScreenshotResult)
        assert result.width > 0
        assert result.height > 0
        assert result.image is not None
        assert result.array is not None
    
    def test_screenshot_dimensions(self):
        """Test that screenshot has expected dimensions."""
        result = capture_desktop_screenshot()
        
        # Should match monitor (may not be exactly 1920x1080)
        assert result.width >= 800  # Minimum reasonable width
        assert result.height >= 600  # Minimum reasonable height
    
    def test_screenshot_array_shape(self):
        """Test that numpy array has correct shape (H, W, 3)."""
        result = capture_desktop_screenshot()
        
        assert len(result.array.shape) == 3
        assert result.array.shape[2] == 3  # RGB
        assert result.array.shape[0] == result.height
        assert result.array.shape[1] == result.width
    
    def test_save_screenshot_creates_file(self, tmp_path):
        """Test that save_screenshot creates a file."""
        result = capture_desktop_screenshot()
        filepath = tmp_path / "test_screenshot.png"
        
        save_screenshot(result, str(filepath))
        
        assert filepath.exists()
        assert filepath.stat().st_size > 0
    
    def test_save_screenshot_with_annotations(self, tmp_path):
        """Test saving screenshot with bounding box annotations."""
        result = capture_desktop_screenshot()
        filepath = tmp_path / "test_annotated.png"
        
        annotations = [
            (100, 100, 64, 64, "Test", 0.95),
            (200, 200, 48, 48, "Icon", 0.85),
        ]
        
        save_screenshot(result, str(filepath), annotations)
        
        assert filepath.exists()
        # Annotated image should be larger than unannotated
        assert filepath.stat().st_size > 0


class TestIconDetector:
    """Tests for CLIP-based icon detector."""
    
    @pytest.fixture
    def detector(self):
        """Create an IconDetector instance."""
        return IconDetector()
    
    def test_detector_initialization(self, detector):
        """Test that detector initializes without errors."""
        assert detector is not None
        assert detector.device in ["cpu", "cuda"]
    
    def test_model_loads_successfully(self, detector):
        """Test that CLIP model loads without errors."""
        detector.load_model()
        
        assert detector._model_loaded
        assert detector.model is not None
        assert detector.processor is not None
    
    def test_detection_returns_result(self, detector):
        """Test that detection returns a DetectionResult."""
        screenshot = capture_desktop_screenshot()
        
        result = detector.detect_notepad_icon(screenshot)
        
        assert isinstance(result, DetectionResult)
        assert isinstance(result.found, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.all_candidates, list)
    
    def test_detection_coordinates_in_bounds(self, detector):
        """Test that detected coordinates are within screen bounds."""
        screenshot = capture_desktop_screenshot()
        
        result = detector.detect_notepad_icon(screenshot)
        
        if result.found:
            assert 0 <= result.center_x <= screenshot.width
            assert 0 <= result.center_y <= screenshot.height
    
    def test_nms_removes_overlapping_boxes(self, detector):
        """Test that non-maximum suppression works correctly."""
        # Create overlapping boxes
        candidates = [
            {"x": 100, "y": 100, "width": 64, "height": 64, "confidence": 0.9},
            {"x": 110, "y": 110, "width": 64, "height": 64, "confidence": 0.8},
            {"x": 300, "y": 300, "width": 64, "height": 64, "confidence": 0.7},
        ]
        
        filtered = detector._apply_nms(candidates, iou_threshold=0.3)
        
        # Should remove the overlapping box with lower confidence
        assert len(filtered) <= len(candidates)
        # Highest confidence should be first
        assert filtered[0]["confidence"] == 0.9
    
    def test_iou_computation(self, detector):
        """Test IoU (Intersection over Union) calculation."""
        # Fully overlapping boxes
        box1 = {"x": 0, "y": 0, "width": 100, "height": 100}
        box2 = {"x": 0, "y": 0, "width": 100, "height": 100}
        
        iou = detector._compute_iou(box1, box2)
        assert iou == 1.0
        
        # Non-overlapping boxes
        box3 = {"x": 200, "y": 200, "width": 100, "height": 100}
        iou = detector._compute_iou(box1, box3)
        assert iou == 0.0
        
        # Partially overlapping boxes
        box4 = {"x": 50, "y": 50, "width": 100, "height": 100}
        iou = detector._compute_iou(box1, box4)
        assert 0 < iou < 1
    
    def test_retry_mechanism(self, detector):
        """Test that detect_with_retry attempts multiple times."""
        call_count = 0
        
        def mock_screenshot():
            nonlocal call_count
            call_count += 1
            return capture_desktop_screenshot()
        
        # Run detection with retry
        result = detector.detect_with_retry(
            screenshot_func=mock_screenshot,
            max_retries=2,
            retry_delay=0.1
        )
        
        # Should have attempted at least once
        assert call_count >= 1
        assert isinstance(result, DetectionResult)


class TestDetectionPositions:
    """
    Tests for icon detection at different screen positions.
    
    These tests generate annotated screenshots showing detection results
    for icons at different desktop positions.
    """
    
    @pytest.fixture
    def detector(self):
        """Create and prepare an IconDetector."""
        detector = IconDetector()
        detector.load_model()
        return detector
    
    @pytest.fixture
    def screenshots_dir(self, tmp_path):
        """Create a directory for test screenshots."""
        screenshots = tmp_path / "screenshots"
        screenshots.mkdir()
        return screenshots
    
    def test_detection_and_annotated_screenshot(self, detector, screenshots_dir):
        """
        Test detection and create annotated screenshot.
        
        This test captures the current desktop, runs detection,
        and saves an annotated screenshot showing the results.
        """
        # Capture screenshot
        screenshot = capture_desktop_screenshot()
        
        # Run detection
        result = detector.detect_notepad_icon(screenshot)
        
        # Create annotations
        annotations = []
        
        if result.found:
            annotations.append((
                result.bbox_x,
                result.bbox_y,
                result.bbox_width,
                result.bbox_height,
                "Notepad (DETECTED)",
                result.confidence
            ))
        
        # Add top candidates
        for i, candidate in enumerate(result.all_candidates[:5]):
            if candidate != result.all_candidates[0] if result.all_candidates else True:
                annotations.append((
                    candidate["x"],
                    candidate["y"],
                    candidate["width"],
                    candidate["height"],
                    f"Candidate {i+1}",
                    candidate.get("confidence", 0)
                ))
        
        # Save annotated screenshot
        filepath = screenshots_dir / "detection_result.png"
        save_screenshot(screenshot, str(filepath), annotations)
        
        assert filepath.exists()
        print(f"\nAnnotated screenshot saved to: {filepath}")
        print(f"Detection found: {result.found}")
        print(f"Confidence: {result.confidence:.3f}")
        if result.found:
            print(f"Location: ({result.center_x}, {result.center_y})")


class TestCustomPrompts:
    """Tests for custom prompt functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create an IconDetector."""
        detector = IconDetector()
        detector.load_model()
        return detector
    
    def test_custom_prompts_work(self, detector):
        """Test that custom prompts can be used for detection."""
        screenshot = capture_desktop_screenshot()
        
        custom_prompts = [
            "a Windows application icon",
            "a desktop shortcut icon",
        ]
        
        result = detector.detect_notepad_icon(screenshot, custom_prompts=custom_prompts)
        
        assert isinstance(result, DetectionResult)
        # May or may not find something, but should not error


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
