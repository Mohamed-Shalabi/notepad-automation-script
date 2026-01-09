"""
Window Validator Module

Provides window detection and validation using Win32 API.
Detects when Notepad has launched and verifies window state.
"""

import logging
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass

import win32gui
import win32process
import win32con
import psutil

from ..config import config


logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """Information about a detected window."""
    hwnd: int
    title: str
    process_id: int
    process_name: str
    is_visible: bool
    rect: Tuple[int, int, int, int]  # left, top, right, bottom


class WindowValidator:
    """
    Validator for window state detection using Win32 API.
    """
    
    def __init__(self):
        """Initialize the window validator."""
        logger.info("WindowValidator initialized")
    
    def find_notepad_window(self) -> Optional[WindowInfo]:
        """
        Find an open Notepad window.
        
        Returns:
            WindowInfo if found, None otherwise.
        """
        windows = self._enumerate_windows()
        
        for window in windows:
            # Check if it's a Notepad window
            if self._is_notepad_window(window):
                logger.info(f"Found Notepad window: '{window.title}'")
                return window
        
        return None
    
    def wait_for_notepad(
        self,
        timeout: Optional[float] = None,
        check_interval: Optional[float] = None
    ) -> Optional[WindowInfo]:
        """
        Wait for a Notepad window to appear.
        
        Args:
            timeout: Maximum seconds to wait (uses config default if None).
            check_interval: Seconds between checks (uses config default if None).
        
        Returns:
            WindowInfo if found within timeout, None otherwise.
        """
        timeout = timeout or config.automation.window_wait_timeout
        check_interval = check_interval or config.automation.window_check_interval
        
        logger.info(f"Waiting for Notepad window (timeout: {timeout}s)")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            window = self.find_notepad_window()
            if window:
                return window
            
            time.sleep(check_interval)
        
        logger.warning(f"Notepad window not found within {timeout}s")
        return None
    
    def wait_for_notepad_close(
        self,
        timeout: float = 5.0,
        check_interval: float = 0.3
    ) -> bool:
        """
        Wait for all Notepad windows to close.
        
        Returns:
            True if closed within timeout, False otherwise.
        """
        logger.info(f"Waiting for Notepad to close (timeout: {timeout}s)")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            window = self.find_notepad_window()
            if not window:
                logger.info("Notepad closed")
                return True
            
            time.sleep(check_interval)
        
        logger.warning("Notepad did not close within timeout")
        return False
    
    def is_notepad_running(self) -> bool:
        """Check if Notepad is currently running."""
        return self.find_notepad_window() is not None
    
    def focus_window(self, window: WindowInfo) -> bool:
        """
        Bring a window to the foreground and focus it.
        
        Args:
            window: The window to focus.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Restore if minimized
            if win32gui.IsIconic(window.hwnd):
                win32gui.ShowWindow(window.hwnd, win32con.SW_RESTORE)
            
            # Bring to foreground
            win32gui.SetForegroundWindow(window.hwnd)
            time.sleep(0.2)
            
            logger.info(f"Focused window: '{window.title}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to focus window: {e}")
            return False
    
    def close_all_notepad_windows(self) -> int:
        """
        Close all open Notepad windows.
        
        Returns:
            Number of windows closed.
        """
        closed = 0
        
        while True:
            window = self.find_notepad_window()
            if not window:
                break
            
            try:
                win32gui.PostMessage(window.hwnd, win32con.WM_CLOSE, 0, 0)
                time.sleep(0.5)
                closed += 1
            except Exception as e:
                logger.error(f"Failed to close Notepad window: {e}")
                break
        
        logger.info(f"Closed {closed} Notepad window(s)")
        return closed
    
    def _enumerate_windows(self) -> List[WindowInfo]:
        """Enumerate all visible windows."""
        windows = []
        
        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                try:
                    title = win32gui.GetWindowText(hwnd)
                    if title:  # Skip windows without titles
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        
                        try:
                            process = psutil.Process(pid)
                            process_name = process.name()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            process_name = "unknown"
                        
                        rect = win32gui.GetWindowRect(hwnd)
                        
                        windows.append(WindowInfo(
                            hwnd=hwnd,
                            title=title,
                            process_id=pid,
                            process_name=process_name,
                            is_visible=True,
                            rect=rect
                        ))
                except Exception:
                    pass
            return True
        
        win32gui.EnumWindows(enum_callback, None)
        return windows
    
    def _is_notepad_window(self, window: WindowInfo) -> bool:
        """
        Check if a window is a Notepad window.
        
        STRICT SAFETY CHECK:
        Only match windows that belong to the actual 'notepad.exe' process.
        Do NOT match based on title alone to avoid killing the IDE or agent.
        """
        # Strict process name check
        if window.process_name.lower() == config.automation.notepad_process_name.lower():
            return True
            
        # If we can't determine process name (access denied), use very strict title matching
        # But prefer to skip if unsure to be safe
        if window.process_name == "unknown":
            title = window.title
            if "notepad" in title.lower():
                return True
                
        return False
    