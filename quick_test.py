"""
Quick Test Script

Runs a minimal test to verify the system is working without
performing actual automation. Good for checking installation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    from notepad_automation.config import config
    print("  [OK] config")
    
    from notepad_automation.grounding.screenshot import capture_desktop_screenshot
    print("  [OK] grounding.screenshot")
    
    from notepad_automation.grounding.detector import IconDetector
    print("  [OK] grounding.detector")
    
    from notepad_automation.automation.mouse import MouseController
    print("  [OK] automation.mouse")
    
    from notepad_automation.automation.keyboard import KeyboardController
    print("  [OK] automation.keyboard")
    
    from notepad_automation.automation.window import WindowValidator
    print("  [OK] automation.window")
    
    from notepad_automation.api.client import APIClient
    print("  [OK] api.client")
    
    from notepad_automation.files.manager import FileManager
    print("  [OK] files.manager")
    
    print("\nAll imports successful!")
    return True


def test_screenshot():
    """Test screenshot capture."""
    print("\nTesting screenshot capture...")
    
    from notepad_automation.grounding.screenshot import capture_desktop_screenshot
    
    result = capture_desktop_screenshot()
    print(f"  [OK] Captured {result.width}x{result.height} screenshot")
    
    return True


def test_api():
    """Test API connection."""
    print("\nTesting API connection...")
    
    from notepad_automation.api.client import APIClient
    
    client = APIClient()
    if client.test_connection():
        print("  [OK] API connection successful")
    else:
        print("  [WARN] API connection failed (may work anyway)")
    
    return True


def test_window_enumeration():
    """Test window enumeration."""
    print("\nTesting window enumeration...")
    
    from notepad_automation.automation.window import WindowValidator
    
    validator = WindowValidator()
    windows = validator._enumerate_windows()
    print(f"  [OK] Found {len(windows)} windows")
    
    notepad = validator.find_notepad_window()
    if notepad:
        print(f"  [OK] Notepad is running: '{notepad.title}'")
    else:
        print("  [INFO] Notepad is not running")
    
    return True


def test_clip_model():
    """Test CLIP model loading."""
    print("\nTesting CLIP model loading (this may take a while)...")
    
    from notepad_automation.grounding.detector import IconDetector
    
    detector = IconDetector()
    print(f"  [INFO] Using device: {detector.device}")
    
    detector.load_model()
    print("  [OK] CLIP model loaded successfully")
    
    return True


def main():
    """Run all quick tests."""
    print("=" * 50)
    print("Notepad Automation - Quick Test")
    print("=" * 50)
    
    try:
        test_imports()
        test_screenshot()
        test_api()
        test_window_enumeration()
        test_clip_model()
        
        print("\n" + "=" * 50)
        print("All tests passed! [OK]")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
