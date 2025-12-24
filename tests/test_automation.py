"""
Tests for Automation Modules

Tests for mouse control, keyboard automation, and window validation.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from notepad_automation.automation.mouse import MouseController
from notepad_automation.automation.keyboard import KeyboardController
from notepad_automation.automation.window import WindowValidator, WindowInfo


class TestMouseController:
    """Tests for MouseController functionality."""
    
    @pytest.fixture
    def mouse(self):
        """Create a MouseController instance."""
        return MouseController()
    
    def test_initialization(self, mouse):
        """Test that mouse controller initializes."""
        assert mouse is not None
    
    def test_get_position(self, mouse):
        """Test getting current mouse position."""
        x, y = mouse.get_position()
        
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert x >= 0
        assert y >= 0
    
    def test_move_to_returns_position(self, mouse):
        """Test that move_to returns the target position."""
        # Move to a safe position
        x, y = mouse.move_to(100, 100, duration=0.1)
        
        assert abs(x - 100) <= 5  # Allow small randomness
        assert abs(y - 100) <= 5


class TestKeyboardController:
    """Tests for KeyboardController functionality."""
    
    @pytest.fixture
    def keyboard(self):
        """Create a KeyboardController instance."""
        return KeyboardController()
    
    def test_initialization(self, keyboard):
        """Test that keyboard controller initializes."""
        assert keyboard is not None


class TestWindowValidator:
    """Tests for WindowValidator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create a WindowValidator instance."""
        return WindowValidator()
    
    def test_initialization(self, validator):
        """Test that window validator initializes."""
        assert validator is not None
    
    def test_enumerate_windows(self, validator):
        """Test that windows can be enumerated."""
        windows = validator._enumerate_windows()
        
        assert isinstance(windows, list)
        # Should find at least some windows
        assert len(windows) > 0
    
    def test_window_info_structure(self, validator):
        """Test that WindowInfo has correct structure."""
        windows = validator._enumerate_windows()
        
        if windows:
            window = windows[0]
            assert isinstance(window, WindowInfo)
            assert hasattr(window, 'hwnd')
            assert hasattr(window, 'title')
            assert hasattr(window, 'process_name')
            assert hasattr(window, 'is_visible')
    
    def test_find_notepad_when_not_running(self, validator):
        """Test finding Notepad when it's not running."""
        # This may or may not return a window depending on system state
        result = validator.find_notepad_window()
        
        # Result should be None or WindowInfo
        assert result is None or isinstance(result, WindowInfo)
    
    def test_is_notepad_running(self, validator):
        """Test checking if Notepad is running."""
        result = validator.is_notepad_running()
        
        assert isinstance(result, bool)


class TestWindowValidation:
    """Integration tests for window validation with actual Notepad."""
    
    @pytest.fixture
    def validator(self):
        """Create a WindowValidator instance."""
        return WindowValidator()
    
    def test_wait_for_notepad_timeout(self, validator):
        """Test that wait_for_notepad respects timeout."""
        start = time.time()
        
        # Use a short timeout
        result = validator.wait_for_notepad(timeout=1.0, check_interval=0.5)
        
        elapsed = time.time() - start
        
        # Should return within reasonable time of timeout
        assert elapsed < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
