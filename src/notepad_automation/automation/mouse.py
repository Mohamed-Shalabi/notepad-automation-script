"""
Mouse Controller Module

Provides precise mouse control for desktop automation, including
movement, clicking, and double-clicking with natural-feeling delays.
"""

import logging
import random
from typing import Tuple, Optional

import pyautogui

from ..config import config


logger = logging.getLogger(__name__)


# Configure pyautogui safety features
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.05  # Small pause between actions


class MouseController:
    """
    Controller for mouse operations with natural movement and safety features.
    """
    
    def __init__(self):
        """Initialize the mouse controller."""
        logger.info("MouseController initialized")
    
    def get_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        return pyautogui.position()
    
    def move_to(
        self,
        x: int,
        y: int,
        duration: Optional[float] = None,
        add_randomness: bool = True
    ) -> Tuple[int, int]:
        """
        Move mouse to specified coordinates.
        
        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            duration: Movement duration in seconds (uses config default if None).
            add_randomness: If True, adds small random offset for natural movement.
        
        Returns:
            Tuple of actual (x, y) position moved to.
        """
        duration = duration or config.automation.mouse_move_duration
        
        # Add small random offset for more natural behavior
        if add_randomness:
            x += random.randint(-2, 2)
            y += random.randint(-2, 2)
        
        # Ensure coordinates are within screen bounds
        screen_width, screen_height = pyautogui.size()
        x = max(0, min(x, screen_width - 1))
        y = max(0, min(y, screen_height - 1))
        
        logger.debug(f"Moving mouse to ({x}, {y}) over {duration}s")
        
        pyautogui.moveTo(x, y, duration=duration)
        
        return (x, y)
    
    def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left"
    ) -> None:
        """
        Click at coordinates (or current position if not specified).
        
        Args:
            x: X coordinate (optional).
            y: Y coordinate (optional).
            button: Mouse button ('left', 'right', 'middle').
        """
        if x is not None and y is not None:
            self.move_to(x, y)
        
        pos = self.get_position()
        logger.debug(f"Clicking {button} button at {pos}")
        
        pyautogui.click(button=button)
    
    def double_click(
        self,
        x: int,
        y: int,
        interval: Optional[float] = None
    ) -> None:
        """
        Double-click at specified coordinates.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            interval: Interval between clicks in seconds.
        """
        interval = interval or config.automation.double_click_interval
        
        logger.info(f"Double-clicking at ({x}, {y})")
        
        self.move_to(x, y)
        pyautogui.doubleClick(interval=interval)
    
    def right_click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None
    ) -> None:
        """Right-click at coordinates (or current position)."""
        self.click(x, y, button="right")
    
    def drag_to(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5
    ) -> None:
        """
        Drag from one position to another.
        
        Args:
            start_x: Starting X coordinate.
            start_y: Starting Y coordinate.
            end_x: Ending X coordinate.
            end_y: Ending Y coordinate.
            duration: Drag duration in seconds.
        """
        logger.debug(f"Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        
        self.move_to(start_x, start_y)
        pyautogui.drag(
            end_x - start_x,
            end_y - start_y,
            duration=duration
        )
