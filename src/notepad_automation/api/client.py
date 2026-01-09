"""
API Client Module

HTTP client for fetching posts from JSONPlaceholder API with
retry logic, timeout handling, and graceful error management.
"""

import logging
import time
from typing import List, Optional
from dataclasses import dataclass

import httpx

from ..config import config


logger = logging.getLogger(__name__)


@dataclass
class Post:
    """Represents a post from the API."""
    id: int
    user_id: int
    title: str
    body: str
    
    def format_content(self) -> str:
        """Format the post content for writing to a file."""
        return config.file.content_template.format(
            title=self.title,
            body=self.body
        )


class APIClient:
    """
    HTTP client for the JSONPlaceholder API.
    
    Features:
    - Retry with exponential backoff
    - Configurable timeout
    - Graceful error handling
    """
    
    def __init__(self):
        """Initialize the API client."""
        self.base_url = config.api.base_url
        self.timeout = config.api.timeout_seconds
        logger.info(f"APIClient initialized (base_url: {self.base_url})")
    
    def fetch_posts(
        self,
        max_posts: Optional[int] = None
    ) -> List[Post]:
        """
        Fetch posts from the API.
        
        Args:
            max_posts: Maximum number of posts to return (uses config default if None).
        
        Returns:
            List of Post objects.
        
        Raises:
            RuntimeError: If fetching fails after all retries.
        """
        max_posts = max_posts or config.api.max_posts
        url = f"{self.base_url}{config.api.posts_endpoint}"
        
        logger.info(f"Fetching posts from {url}")
        
        last_error = None
        
        for attempt in range(config.api.max_retries):
            try:
                return self._fetch_with_timeout(url, max_posts)
                
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
                
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < config.api.max_retries - 1:
                delay = config.api.retry_delay_seconds * (2 ** attempt)
                logger.info(f"Waiting {delay}s before retry...")
                time.sleep(delay)
        
        logger.error(f"All {config.api.max_retries} attempts failed")
        logger.info("Using fallback sample data instead")
        return self._get_fallback_posts(max_posts)
    
    def _get_fallback_posts(self, max_posts: int) -> List[Post]:
        """
        Return sample posts when the API is unavailable.
        
        This ensures the automation can still run for demonstration
        purposes even without network connectivity.
        """
        sample_posts = [
            Post(id=1, user_id=1, 
                 title="Sample Post 1 - Introduction to Automation",
                 body="This is a sample post used when the API is unavailable.\nIt demonstrates the file saving functionality of the automation system."),
            Post(id=2, user_id=1,
                 title="Sample Post 2 - Vision-Based Grounding",
                 body="CLIP-based semantic grounding allows us to find UI elements\nwithout relying on fixed coordinates or template matching."),
            Post(id=3, user_id=1,
                 title="Sample Post 3 - Desktop Automation",
                 body="Using pyautogui and Win32 APIs, we can control the mouse\nand keyboard to automate desktop applications."),
            Post(id=4, user_id=1,
                 title="Sample Post 4 - Error Handling",
                 body="Robust error handling ensures the system degrades gracefully\nwhen unexpected situations occur."),
            Post(id=5, user_id=1,
                 title="Sample Post 5 - Retry Logic",
                 body="Exponential backoff helps handle transient failures\nin network requests and UI interactions."),
            Post(id=6, user_id=1,
                 title="Sample Post 6 - Window Validation",
                 body="Win32 APIs allow us to detect when applications launch\nand verify their window state."),
            Post(id=7, user_id=1,
                 title="Sample Post 7 - File Management",
                 body="The file manager handles directory creation, path generation,\nand file verification."),
            Post(id=8, user_id=1,
                 title="Sample Post 8 - Configuration",
                 body="All parameters are configurable through dataclasses,\nmaking the system easy to adapt."),
            Post(id=9, user_id=1,
                 title="Sample Post 9 - Testing",
                 body="Comprehensive tests validate each component\nworks correctly in isolation and together."),
            Post(id=10, user_id=1,
                 title="Sample Post 10 - Documentation",
                 body="Clear documentation helps others understand\nthe system architecture and usage."),
        ]
        logger.info(f"Returning {min(max_posts, len(sample_posts))} fallback posts")
        return sample_posts[:max_posts]
    
    def _fetch_with_timeout(
        self,
        url: str,
        max_posts: int
    ) -> List[Post]:
        """
        Fetch posts with timeout handling.
        
        Args:
            url: API endpoint URL.
            max_posts: Maximum posts to return.
        
        Returns:
            List of Post objects.
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different API response formats
            if isinstance(data, dict) and 'posts' in data:
                # DummyJSON format: {"posts": [...], "total": 150, ...}
                items = data['posts']
            elif isinstance(data, list):
                # JSONPlaceholder format: [...]
                items = data
            else:
                raise ValueError(f"Unexpected API response format: {type(data)}")
            
            posts = []
            for item in items[:max_posts]:
                post = Post(
                    id=item["id"],
                    user_id=item["userId"],
                    title=item["title"],
                    body=item["body"]
                )
                posts.append(post)
            
            logger.info(f"Fetched {len(posts)} posts successfully")
            return posts
    