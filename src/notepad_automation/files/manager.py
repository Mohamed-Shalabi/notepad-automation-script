"""
File Manager Module

Manages file and directory operations for saving post content.
Handles directory creation, file path generation, and validation.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from ..config import config


logger = logging.getLogger(__name__)


class FileManager:
    """
    Manager for file operations related to saving post content.
    """
    
    def __init__(self):
        """Initialize the file manager."""
        self.output_dir = config.file.output_directory
        logger.info(f"FileManager initialized (output: {self.output_dir})")
    
    def ensure_output_directory(self) -> Path:
        """
        Ensure the output directory exists.
        
        Creates the directory if it doesn't exist.
        
        Returns:
            Path to the output directory.
        """
        if not self.output_dir.exists():
            logger.info(f"Creating output directory: {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.debug(f"Output directory exists: {self.output_dir}")
        
        return self.output_dir
    
    def get_file_path(self, post_id: int) -> Path:
        """
        Get the full file path for a post.
        
        Args:
            post_id: The post ID.
        
        Returns:
            Full path to the file.
        """
        return config.file.get_file_path(post_id)
    
    def file_exists(self, post_id: int) -> bool:
        """
        Check if a file for a post already exists.
        
        Args:
            post_id: The post ID.
        
        Returns:
            True if file exists, False otherwise.
        """
        return self.get_file_path(post_id).exists()
    
    def verify_file_saved(
        self,
        post_id: int,
        expected_content: Optional[str] = None,
        timeout: float = 5.0
    ) -> bool:
        """
        Verify that a file was saved correctly.
        
        Args:
            post_id: The post ID.
            expected_content: Optional content to verify (if None, just checks existence).
            timeout: Maximum seconds to wait for the file to appear.
        
        Returns:
            True if file exists (and matches content if provided), False otherwise.
        """
        import time
        
        file_path = self.get_file_path(post_id)
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if file_path.exists():
                if expected_content is None:
                    logger.info(f"File verified: {file_path}")
                    return True
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if expected_content in content:
                        logger.info(f"File content verified: {file_path}")
                        return True
                    else:
                        logger.warning(f"File content mismatch: {file_path}")
                except Exception as e:
                    logger.warning(f"Error reading file: {e}")
            
            time.sleep(0.5)
        
        logger.warning(f"File verification failed: {file_path}")
        return False
    
    def cleanup_existing_files(self) -> int:
        """
        Remove existing post files from the output directory.
        
        Returns:
            Number of files removed.
        """
        removed = 0
        
        if not self.output_dir.exists():
            return 0
        
        for file in self.output_dir.glob(f"{config.file.file_prefix}*{config.file.file_extension}"):
            try:
                file.unlink()
                removed += 1
                logger.debug(f"Removed: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove {file}: {e}")
        
        logger.info(f"Cleaned up {removed} existing file(s)")
        return removed
    
    def get_output_directory_str(self) -> str:
        """Get the output directory as a string path."""
        return str(self.output_dir)
    
    def list_saved_files(self) -> list:
        """
        List all saved post files.
        
        Returns:
            List of file paths.
        """
        if not self.output_dir.exists():
            return []
        
        files = list(self.output_dir.glob(
            f"{config.file.file_prefix}*{config.file.file_extension}"
        ))
        return sorted(files, key=lambda f: f.name)
    
    def get_summary(self) -> dict:
        """
        Get a summary of the file manager state.
        
        Returns:
            Dictionary with output directory info and file counts.
        """
        files = self.list_saved_files()
        return {
            "output_directory": str(self.output_dir),
            "directory_exists": self.output_dir.exists(),
            "file_count": len(files),
            "files": [str(f.name) for f in files]
        }
