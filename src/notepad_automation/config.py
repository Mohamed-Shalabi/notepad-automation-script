"""
Configuration constants for the Notepad Automation System.

This module centralizes all configurable parameters to make the system
easy to tune and adapt to different environments.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class ScreenConfig:
    """Screen and display configuration."""
    width: int = 1920
    height: int = 1080
    primary_monitor: int = 1  # 1-indexed for mss


@dataclass
class GroundingConfig:
    """Vision grounding configuration."""
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 0.5
    
    # Detection thresholds
    confidence_threshold: float = 0.50  # Probability-based threshold with Negative Prompting
    nms_iou_threshold: float = 0.3  # Non-maximum suppression IoU threshold
    
    # Sliding window settings
    window_sizes: List[int] = field(default_factory=lambda: [40, 64, 100])
    window_stride: int = 32  # Pixels to move between windows
    
    # Icon size constraints (pixels)
    min_icon_size: int = 32
    max_icon_size: int = 128
    
    # CLIP model
    clip_model_name: str = "openai/clip-vit-base-patch32"
    
    # Text prompts for Notepad icon detection
    notepad_prompts: List[str] = field(default_factory=lambda: [
        "a Windows desktop icon for the Notepad application",
        "a Notepad shortcut icon with a signature notebook",
        "a desktop launcher for Notepad with text",
        "a small notebook icon on a computer desktop",
        "a Notepad app icon, not just plain text",
    ])
    
    # Negative prompts to suppress false positives (text, wallpaper, etc)
    negative_prompts: List[str] = field(default_factory=lambda: [
        "plain text not containing \"Notepad\"",
        "plain text containing \"Notebook\" but in a paragraph of text",
        "icon not related to Notepad",
        "a folder icon",
        "desktop wallpaper texture",
        "random user interface elements",
    ])


@dataclass
class AutomationConfig:
    """Mouse and keyboard automation configuration."""
    # Mouse settings
    double_click_interval: float = 0.1  # Seconds between clicks
    mouse_move_duration: float = 0.2  # Seconds to move mouse
    
    # Keyboard settings
    typing_interval: float = 0.02  # Seconds between keystrokes
    
    # Window detection
    window_wait_timeout: float = 10.0  # Max seconds to wait for window
    window_check_interval: float = 0.5  # Seconds between window checks
    
    # Notepad window identifiers
    notepad_window_titles: List[str] = field(default_factory=lambda: [
        "Untitled - Notepad",
        "Notepad",
        "*Untitled - Notepad",
    ])
    notepad_process_name: str = "notepad.exe"


@dataclass
class APIConfig:
    """API configuration settings."""
    base_url: str = "https://dummyjson.com"
    posts_endpoint: str = "/posts"
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    timeout_seconds: float = 10.0
    max_posts: int = 10


@dataclass
class FileConfig:
    """File management configuration."""
    # Output directory (relative to Desktop)
    output_folder_name: str = "tjm-project"
    file_prefix: str = "post_"
    file_extension: str = ".txt"
    
    # File content format
    content_template: str = "Title: {title}\n\n{body}"
    
    @property
    def output_directory(self) -> Path:
        """Get the full path to the output directory on Desktop."""
        desktop = Path(os.path.expanduser("~")) / "Desktop"
        return desktop / self.output_folder_name
    
    def get_file_path(self, post_id: int) -> Path:
        """Generate file path for a specific post."""
        filename = f"{self.file_prefix}{post_id}{self.file_extension}"
        return self.output_directory / filename


@dataclass
class LogConfig:
    """Logging configuration."""
    log_directory: Path = field(default_factory=lambda: Path("logs"))
    log_filename: str = "automation.log"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @property
    def log_file_path(self) -> Path:
        """Get full path to the log file."""
        return self.log_directory / self.log_filename


@dataclass
class Config:
    """Master configuration combining all sub-configurations."""
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    file: FileConfig = field(default_factory=FileConfig)
    log: LogConfig = field(default_factory=LogConfig)


# Global configuration instance
config = Config()
