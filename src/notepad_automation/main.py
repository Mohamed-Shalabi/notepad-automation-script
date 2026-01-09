"""
Main Orchestrator Module

This is the entry point for the Notepad Automation System.
Orchestrates the complete workflow:

1. Fetch posts from API
2. For each post:
   a. Capture desktop screenshot
   b. Detect Notepad icon using CLIP
   c. Double-click to launch Notepad
   d. Verify window appeared
   e. Type post content
   f. Save file with correct filename
   g. Close Notepad
3. Report results

The system includes comprehensive error handling, retries,
and detailed logging throughout the process.
"""

import logging
import sys
import time
import cv2
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from .config import config
from .automation import MouseController, KeyboardController, WindowValidator
from .api import APIClient, Post
from .files import FileManager
from .find_icon_coordinates.find_icon_coordinates import find_icon_coordinates
import pyautogui


# Configure logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging for the application."""
    # Ensure log directory exists
    config.log.log_directory.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("notepad_automation")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    level = getattr(logging, log_level.upper())
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(
        config.log.log_file_path,
        mode='w',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(config.log.log_format)
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


@dataclass
class AutomationResult:
    """Result of a single post automation."""
    post_id: int
    success: bool
    file_path: Optional[str]
    error: Optional[str]
    duration_seconds: float


@dataclass
class RunResult:
    """Result of the complete automation run."""
    success: bool
    total_posts: int
    successful_posts: int
    failed_posts: int
    results: List[AutomationResult]
    total_duration_seconds: float


class NotepadAutomation:
    """
    Main orchestrator for the Notepad automation workflow.
    
    This class coordinates all components to:
    1. Fetch data from the API
    2. Detect the Notepad icon on the desktop
    3. Automate Notepad to create and save files
    """
    
    def __init__(self):
        """Initialize the automation system."""
        self.logger = logging.getLogger("notepad_automation.main")
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.window = WindowValidator()
        self.api = APIClient()
        self.files = FileManager()
        
        self.logger.info("NotepadAutomation initialized")
    
    def run(self) -> RunResult:
        """
        Execute the complete automation workflow.
        
        Returns:
            RunResult with details of the automation run.
        """
        start_time = time.time()
        results: List[AutomationResult] = []
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Notepad Automation System")
        self.logger.info("=" * 60)
        
        # Step 0: Ensure prerequisites
        try:
            self._ensure_prerequisites()
        except Exception as e:
            self.logger.error(f"Prerequisites check failed: {e}")
            return RunResult(
                success=False,
                total_posts=0,
                successful_posts=0,
                failed_posts=0,
                results=[],
                total_duration_seconds=time.time() - start_time
            )
        
        # Step 1: Fetch posts from API
        self.logger.info("-" * 40)
        self.logger.info("Step 1: Fetching posts from API")
        
        try:
            posts = self.api.fetch_posts(max_posts=config.api.max_posts)
            self.logger.info(f"Successfully fetched {len(posts)} posts")
        except Exception as e:
            self.logger.error(f"Failed to fetch posts: {e}")
            return RunResult(
                success=False,
                total_posts=0,
                successful_posts=0,
                failed_posts=0,
                results=[],
                total_duration_seconds=time.time() - start_time,
                screenshots_saved=self.saved_screenshots
            )
        
        # Step 2: Ensure output directory exists
        self.files.ensure_output_directory()
        
        # Step 3: Close any existing Notepad windows
        self.window.close_all_notepad_windows()
        time.sleep(0.5)

        # # Step 4: Detect Notepad icon
        # self.logger.info("Minimizing all windows to show desktop...")
        # pyautogui.hotkey('win', 'd')
        # time.sleep(1.0) # wait for animation
        
        self.logger.info("Capturing desktop screenshot and searching for Notepad icon...")
        
        # Ensure screenshots directory exists
        Path("screenshots").mkdir(exist_ok=True)
        
        screenshot_path = "screenshots/desktop_current.png"
        pyautogui.screenshot(screenshot_path)
        
        icons_dir = str(Path(__file__).parent / "supported_icons")
        result = find_icon_coordinates(screenshot_path, icons_dir)
        
        if not result:
            self.logger.error("Notepad icon not detected on desktop using any supported icon template")
            return RunResult(
                success=False,
                total_posts=0,
                successful_posts=0,
                failed_posts=0,
                results=[],
                total_duration_seconds=time.time() - start_time
            )
            
        coords, bbox, icon_name = result
        icon_x, icon_y = coords
        self.logger.info(f"Notepad icon found at ({icon_x}, {icon_y}) using template: {icon_name}")

        # Step 4.1: Annotate and save detection
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            annotated_dir = Path("screenshots/annotated")
            annotated_dir.mkdir(parents=True, exist_ok=True)
            
            annotated_path = annotated_dir / f"annotated_at_{timestamp}.png"
            
            # Read the screenshot
            img = cv2.imread(screenshot_path)
            if img is not None:
                # Draw the box (x, y, w, h)
                bx, by, bw, bh = bbox
                # Green box with thickness 2
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                # Label with icon name
                label = f"Match: {icon_name}"
                cv2.putText(img, label, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imwrite(str(annotated_path), img)
                self.logger.info(f"Annotated screenshot saved to: {annotated_path}")
        except Exception as e:
            self.logger.warning(f"Failed to create annotated screenshot: {e}")
            
        # Step 5: Double-click the icon
        self.logger.info(f"Double-clicking at ({icon_x}, {icon_y})")
        self.mouse.double_click(icon_x, icon_y)
            
        # Step 6: Wait for Notepad window
        window = self.window.wait_for_notepad()
            
        if not window:
            return RunResult(
                success=False,
                total_posts=0,
                successful_posts=0,
                failed_posts=0,
                results=[],
                total_duration_seconds=time.time() - start_time
            )
            
        # Focus the window
        self.window.focus_window(window)
        time.sleep(0.5)
            
        
        # Step 7: Process each post
        for i, post in enumerate(posts):
            self.logger.info("-" * 40)
            self.logger.info(f"Processing post {i + 1}/{len(posts)} (ID: {post.id})")
            
            result = self._process_post(post, i)
            results.append(result)
            
            if result.success:
                self.logger.info(f"Post {post.id} saved successfully")
            else:
                self.logger.error(f"Post {post.id} failed: {result.error}")
            
            # Small delay between posts
            time.sleep(0.5)
        
        # Calculate summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_duration = time.time() - start_time
        
        self.logger.info("=" * 60)
        self.logger.info("Automation Complete")
        self.logger.info(f"Total posts: {len(posts)}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Duration: {total_duration:.1f}s")
        self.logger.info("=" * 60)
        
        return RunResult(
            success=failed == 0,
            total_posts=len(posts),
            successful_posts=successful,
            failed_posts=failed,
            results=results,
            total_duration_seconds=total_duration
        )
    
    def _ensure_prerequisites(self) -> None:
        """Check and ensure all prerequisites are met."""
        self.logger.info("Checking prerequisites...")
        
        # Test API connection
        if not self.api.test_connection():
            self.logger.warning("API connection test failed (may work anyway)")
    
    def _process_post(self, post: Post, index: int) -> AutomationResult:
        """
        Process a single post through the complete workflow.
        
        Args:
            post: The post to process.
            index: Index of the post in the list.
        
        Returns:
            AutomationResult for this post.
        """
        start_time = time.time()
        
        try:
            # Step C.1: strict clean state - Open New File (Ctrl+N)
            # This prevents appending to previous file if Notepad restores session
            self.logger.info("Opening new file (Ctrl+N) to ensure clean buffer")
            self.keyboard.new_file()
            time.sleep(0.5)  # Wait for new tab/window notification/animation
            
            # Step D: Type the content
            content = post.format_content()
            self.logger.info(f"Typing content ({len(content)} chars)")
            self.keyboard.type_text(content, interval=0.05)
            
            # Step E: Save the file
            file_path = self.files.get_file_path(post.id)
            filename = f"post_{post.id}.txt"
            folder = self.files.get_output_directory_str()
            
            self.logger.info(f"Saving to: {file_path}")
            
            # Ctrl+Shift+S to open Save As dialog (always opens dialog)
            # This handles cases where Notepad restores previous session/file
            self.keyboard.save_as()
            time.sleep(0.5)
            
            # Type full absolute path into filename field
            # This avoids the need for flaky folder navigation steps
            self.keyboard.type_filename_and_save(str(file_path), folder_path=None)
            time.sleep(0.5)
            
            # Handle potential overwrite prompt
            self.keyboard.handle_overwrite_prompt(overwrite=True)
            time.sleep(0.5)
            
            # Additional safety wait
            time.sleep(0.5)
            
            # Step F: Verify file was saved
            if not self.files.verify_file_saved(post.id, timeout=3.0):
                return AutomationResult(
                    post_id=post.id,
                    success=False,
                    file_path=str(file_path),
                    error="File verification failed",
                    duration_seconds=time.time() - start_time
                )
            
            return AutomationResult(
                post_id=post.id,
                success=True,
                file_path=str(file_path),
                error=None,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.exception(f"Error processing post {post.id}")
            
            # Try to close any open Notepad windows safely
            try:
                self.window.close_all_notepad_windows()
            except Exception:
                pass
            
            return AutomationResult(
                post_id=post.id,
                success=False,
                file_path=None,
                error=str(e),
                duration_seconds=time.time() - start_time
            )
    


def main():
    """Main entry point for the automation system."""
    # Set up logging
    logger = setup_logging("DEBUG")
    
    try:
        # Create and run automation
        automation = NotepadAutomation()
        result = automation.run()
        
        # Exit with appropriate code
        if result.success:
            logger.info("Automation completed successfully!")
            sys.exit(0)
        else:
            logger.error("Automation completed with failures")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Automation interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
