"""
Automation Module

Provides mouse control, keyboard automation, and window state validation
for desktop automation tasks.
"""

from .mouse import MouseController
from .keyboard import KeyboardController
from .window import WindowValidator

__all__ = ["MouseController", "KeyboardController", "WindowValidator"]
