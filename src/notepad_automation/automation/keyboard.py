"""
Keyboard Controller Module

Provides keyboard automation including typing, special key combinations,
and handling of dialogs and menus.
"""

import logging
import time
from typing import Optional

import pyautogui
import pyperclip

from ..config import config


logger = logging.getLogger(__name__)


class KeyboardController:
    """
    Controller for keyboard operations with proper timing and special key support.
    """
    
    def __init__(self):
        """Initialize the keyboard controller."""
        logger.info("KeyboardController initialized")
    
    def type_text(
        self,
        text: str
    ) -> None:
        """
        Paste text using the system clipboard.
        
        Args:
            text: The text to paste.
        """
        logger.info(f"Pasting {len(text)} characters via clipboard")
        logger.debug(f"Text preview: {text[:50]}...")
        
        self.paste_text(text)

    def paste_text(self, text: str) -> None:
        """
        Copy text to clipboard and paste it.
        
        Args:
            text: The text to paste.
        """
        pyperclip.copy(text)
        time.sleep(0.1)  # Brief wait for clipboard sync
        self.hotkey('ctrl', 'v')
        time.sleep(0.1)  # Brief wait after paste
    
    
    def press_key(self, key: str) -> None:
        """
        Press a single key.
        
        Args:
            key: Key name (e.g., 'enter', 'tab', 'escape').
        """
        logger.debug(f"Pressing key: {key}")
        pyautogui.press(key)
    
    def hotkey(self, *keys: str) -> None:
        """
        Press a key combination.
        
        Args:
            *keys: Keys to press together (e.g., 'ctrl', 's' for Ctrl+S).
        """
        logger.debug(f"Pressing hotkey: {'+'.join(keys)}")
        pyautogui.hotkey(*keys)
    
    def save_file(self) -> None:
        """Send Ctrl+S to save the current file."""
        logger.info("Sending save command (Ctrl+S)")
        self.hotkey('ctrl', 's')
        time.sleep(2.0)  # Increased wait for save dialog (was 0.5s)
    
    def save_as(self) -> None:
        """Send Ctrl+Shift+S for Save As dialog."""
        logger.info("Sending Save As command (Ctrl+Shift+S)")
        self.hotkey('ctrl', 'shift', 's')
        time.sleep(2.0)
    
    def close_window(self) -> None:
        """Send Alt+F4 to close the current window."""
        logger.info("Sending close command (Alt+F4)")
        self.hotkey('alt', 'F4')
        time.sleep(0.5)
    
    def select_all(self) -> None:
        """Send Ctrl+A to select all."""
        logger.debug("Selecting all (Ctrl+A)")
        self.hotkey('ctrl', 'a')
        time.sleep(0.2)
    
    def new_file(self) -> None:
        """Send Ctrl+N for new file."""
        logger.debug("New file (Ctrl+N)")
        self.hotkey('ctrl', 'n')
        time.sleep(0.5)
    
    def cancel_dialog(self) -> None:
        """Press Escape to cancel current dialog."""
        logger.debug("Canceling dialog (Escape)")
        self.press_key('escape')
    
    def confirm_dialog(self) -> None:
        """Press Enter to confirm current dialog."""
        logger.debug("Confirming dialog (Enter)")
        self.press_key('enter')
    
    def navigate_to_folder(self, folder_path: str) -> None:
        """
        Navigate to a folder in a file dialog.
        
        This types the path into the filename field and handles navigation.
        
        Args:
            folder_path: Full path to the folder.
        """
        logger.info(f"Navigating to folder: {folder_path}")
        
        # In Save dialog, Ctrl+L opens location bar
        self.hotkey('ctrl', 'l')
        time.sleep(0.8)  # Wait for focus
        
        # Clear current path and type new one
        self.select_all()
        self.type_text(folder_path)
        time.sleep(0.2)
        self.press_key('enter')
        time.sleep(1.5)  # Wait for folder navigation to complete
    
    def type_filename_and_save(
        self,
        filename: str,
        folder_path: Optional[str] = None
    ) -> None:
        """
        Type a filename in the Save dialog and save.
        
        Args:
            filename: The filename to save as.
            folder_path: Optional folder path to navigate to first.
        """
        logger.info(f"Saving file as: {filename}")
        
        if folder_path:
            self.navigate_to_folder(folder_path)
        
        # Wait for any folder navigation to complete
        time.sleep(0.5)
        
        # Focus filename field (Alt+N in standard dialogs)
        self.hotkey('alt', 'n')
        time.sleep(0.5)
        
        # Clear and type filename
        self.select_all()
        self.type_text(filename)
        time.sleep(0.5)
        
        # Press Save button (Alt+S or Enter)
        self.press_key('enter')
        time.sleep(0.5)
    
    def handle_overwrite_prompt(self, overwrite: bool = True) -> None:
        """
        Handle "file already exists" overwrite prompt.
        
        Args:
            overwrite: If True, confirm overwrite. If False, cancel.
        """
        logger.info(f"Handling overwrite prompt: {'Yes' if overwrite else 'No'}")
        
        if overwrite:
            # Press Yes button (Alt+Y or just Y in most dialogs)
            self.hotkey('alt', 'y')
        else:
            self.press_key('escape')
        
        time.sleep(0.3)
    
    def dismiss_dont_save(self) -> None:
        """
        Handle the "Do you want to save?" prompt by selecting Don't Save.
        """
        logger.info("Dismissing save prompt (Don't Save)")
        # In Windows Notepad, Tab to "Don't Save" and press Enter
        # Or use Alt+N for "Don't Save"
        self.hotkey('alt', 'n')
        time.sleep(0.3)
