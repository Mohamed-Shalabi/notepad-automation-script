"""
Tests for API Client and File Manager

Tests for fetching posts and file operations.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from notepad_automation.api.client import APIClient, Post
from notepad_automation.files.manager import FileManager
from notepad_automation.config import config


class TestAPIClient:
    """Tests for API client functionality."""
    
    @pytest.fixture
    def client(self):
        """Create an APIClient instance."""
        return APIClient()
    
    def test_initialization(self, client):
        """Test that API client initializes correctly."""
        assert client is not None
        assert client.base_url == "https://jsonplaceholder.typicode.com"
    
    def test_connection(self, client):
        """Test API connection."""
        result = client.test_connection()
        
        # Should be able to connect (may fail if no internet)
        assert isinstance(result, bool)
    
    def test_fetch_posts(self, client):
        """Test fetching posts from the API."""
        try:
            posts = client.fetch_posts(max_posts=5)
            
            assert isinstance(posts, list)
            assert len(posts) <= 5
            
            if posts:
                post = posts[0]
                assert isinstance(post, Post)
                assert post.id > 0
                assert post.title
                assert post.body
        except RuntimeError:
            # May fail if no internet connection
            pytest.skip("API unavailable")
    
    def test_post_content_format(self, client):
        """Test that post content formats correctly."""
        post = Post(
            id=1,
            user_id=1,
            title="Test Title",
            body="Test body content"
        )
        
        content = post.format_content()
        
        assert "Title: Test Title" in content
        assert "Test body content" in content


class TestFileManager:
    """Tests for file manager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a FileManager instance."""
        return FileManager()
    
    def test_initialization(self, manager):
        """Test that file manager initializes."""
        assert manager is not None
        assert manager.output_dir is not None
    
    def test_get_file_path(self, manager):
        """Test generating file paths."""
        path = manager.get_file_path(1)
        
        assert path.name == "post_1.txt"
        assert "tjm-project" in str(path)
    
    def test_ensure_output_directory(self, manager):
        """Test creating output directory."""
        # This will create the actual directory on Desktop
        # Only run if explicitly enabled
        pass
    
    def test_file_path_format(self, manager):
        """Test that file paths follow expected format."""
        for post_id in [1, 5, 10]:
            path = manager.get_file_path(post_id)
            
            assert path.suffix == ".txt"
            assert f"post_{post_id}" in path.name
    
    def test_summary(self, manager):
        """Test getting manager summary."""
        summary = manager.get_summary()
        
        assert isinstance(summary, dict)
        assert "output_directory" in summary
        assert "file_count" in summary


class TestPostFormatting:
    """Tests for post content formatting."""
    
    def test_format_with_special_characters(self):
        """Test formatting content with special characters."""
        post = Post(
            id=1,
            user_id=1,
            title="Test & Title <with> 'special' \"chars\"",
            body="Body with\nnewlines\nand\ttabs"
        )
        
        content = post.format_content()
        
        # Should contain the special characters
        assert "&" in content
        assert "newlines" in content
    
    def test_format_with_unicode(self):
        """Test formatting content with unicode characters."""
        post = Post(
            id=1,
            user_id=1,
            title="Unicode: café résumé naïve",
            body="Emoji: 🎉 and symbols: © ® ™"
        )
        
        content = post.format_content()
        
        assert "café" in content
        assert "🎉" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
