"""
Detection Demo Script

This script demonstrates the vision grounding capability by:
1. Capturing a screenshot of the desktop
2. Running CLIP-based detection
3. Saving an annotated screenshot showing results

Run this to generate test screenshots for interview demonstration.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from notepad_automation.config import config
from notepad_automation.grounding.screenshot import (
    capture_desktop_screenshot,
    save_screenshot
)
from notepad_automation.grounding.detector import IconDetector


def demonstrate_detection():
    """
    Run detection and save annotated screenshots.
    
    This creates annotated screenshots showing:
    - The detected Notepad icon (if found)
    - Top candidate region
    - Confidence scores
    """
    print("=" * 60)
    print("Notepad Icon Detection Demo")
    print("=" * 60)
    
    # Create screenshots directory
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    
    # Initialize detector
    print("\n[1/4] Loading CLIP model...")
    detector = IconDetector()
    detector.load_model()
    print("      Model loaded successfully!")
    
    # Capture screenshot
    print("\n[2/4] Capturing desktop screenshot...")
    screenshot = capture_desktop_screenshot()
    print(f"      Screenshot: {screenshot.width}x{screenshot.height}")
    
    # Run detection
    print("\n[3/4] Running CLIP-based icon detection...")
    start_time = time.time()
    result = detector.detect_notepad_icon(screenshot)
    detection_time = time.time() - start_time
    print(f"      Detection completed in {detection_time:.2f}s")
    
    # Report results
    print("\n[4/4] Results:")
    if result.found:
        print(f"      [OK] Notepad icon DETECTED!")
        print(f"      [OK] Location: ({result.center_x}, {result.center_y})")
        print(f"      [OK] Confidence: {result.confidence:.3f}")
        print(f"      [OK] Bounding box: ({result.bbox_x}, {result.bbox_y}, "
              f"{result.bbox_width}x{result.bbox_height})")
    else:
        print("      [X] Notepad icon NOT found")
        print(f"      [X] Best confidence: {result.confidence:.3f}")
        print(f"        (threshold: {config.grounding.confidence_threshold})")
    
    # Show top candidates
    print(f"\n      Top {min(5, len(result.all_candidates))} candidates:")
    for i, candidate in enumerate(result.all_candidates[:5]):
        print(f"        {i+1}. Confidence: {candidate['confidence']:.3f} "
              f"at ({candidate['center_x']}, {candidate['center_y']})")
    
    # Create annotations
    annotations = []
    
    # Main detection (if found)
    if result.found:
        annotations.append((
            result.bbox_x,
            result.bbox_y,
            result.bbox_width,
            result.bbox_height,
            "NOTEPAD",
            result.confidence
        ))
    
    # Top candidates
    for i, candidate in enumerate(result.all_candidates[1:6]):  # Skip first if it's the main detection
        conf = candidate.get("confidence", 0)
        if conf > config.grounding.confidence_threshold * 0.3:
            annotations.append((
                candidate["x"],
                candidate["y"],
                candidate["width"],
                candidate["height"],
                f"#{i+2}",
                conf
            ))
    
    # Save annotated screenshot
    output_path = screenshots_dir / "detection_demo.png"
    save_screenshot(screenshot, str(output_path), annotations)
    print(f"\n      Annotated screenshot saved to: {output_path}")
    
    # Save raw screenshot too
    raw_path = screenshots_dir / "desktop_raw.png"
    save_screenshot(screenshot, str(raw_path), [])
    print(f"      Raw screenshot saved to: {raw_path}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    
    return result


def test_multiple_positions():
    """
    Instructions for testing detection at different positions.
    
    To generate the required screenshots:
    1. Move the Notepad icon to the TOP-LEFT of your desktop
    2. Run: python demo.py --position top_left
    3. Move the Notepad icon to the CENTER of your desktop
    4. Run: python demo.py --position center
    5. Move the Notepad icon to the BOTTOM-RIGHT of your desktop
    6. Run: python demo.py --position bottom_right
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Detection position test")
    parser.add_argument("--position", choices=["top_left", "center", "bottom_right", "demo"],
                        default="demo", help="Position name for the screenshot")
    args = parser.parse_args()
    
    print(f"\nRunning detection for position: {args.position}")
    
    # Initialize
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    
    detector = IconDetector()
    detector.load_model()
    
    # Capture and detect
    screenshot = capture_desktop_screenshot()
    result = detector.detect_notepad_icon(screenshot)
    
    # Create annotations
    annotations = []
    if result.found:
        annotations.append((
            result.bbox_x,
            result.bbox_y,
            result.bbox_width,
            result.bbox_height,
            "NOTEPAD",
            result.confidence
        ))
    
    # Add some candidates
    for i, candidate in enumerate(result.all_candidates[1:4]):
        annotations.append((
            candidate["x"],
            candidate["y"],
            candidate["width"],
            candidate["height"],
            "",
            candidate.get("confidence", 0)
        ))
    
    # Save
    output_path = screenshots_dir / f"detection_{args.position}.png"
    save_screenshot(screenshot, str(output_path), annotations)
    
    print(f"Saved: {output_path}")
    print(f"Found: {result.found}, Confidence: {result.confidence:.3f}")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and "--position" in sys.argv[1]:
        test_multiple_positions()
    else:
        demonstrate_detection()
