# Notepad Automation System

A **production-ready Windows desktop automation system** that uses **CV-based vision grounding** to detect and interact with the Notepad desktop icon using ORB feature matching and multi-scale template matching.

## 🎯 Overview

This system demonstrates **visual UI grounding** - a technique that uses computer vision algorithms to find UI elements based on visual features and patterns, rather than fixed coordinates.

### Key Features

- **CV-Based Grounding**: Uses ORB (Oriented FAST and Rotated BRIEF) feature matching for robust icon detection.
- **Prioritized Multi-Icon Support**: Supports multiple icon templates (e.g., `1.png`, `2.png`) stored in a `supported_icons` directory, matched in numerical priority order.
- **Multi-Scale Fallback**: Automatically falls back to multi-scale template matching if ORB fails, ensuring high reliability across different resolutions and scales.
- **Detection Annotation**: Automatically highlights the detected icon in a screenshot for visual verification.
- **Full Automation**: Fetches data from API, creates files in Notepad, types content, and saves them automatically.
- **Process-Aware**: Intelligent window management that targets specific processes to ensure safety and accuracy.
- **Detailed Logging**: Comprehensive audit trail with debug-level logging of CV matching scores and coordinates.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Orchestrator                         │
│  (Coordinates the complete workflow)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
       ┌──────────────┼──────────────┬─────────────┐
       ▼              ▼              ▼             ▼
┌─────────────┐ ┌───────────┐ ┌──────────┐ ┌────────────┐
│    Vision   │ │Automation │ │   API    │ │   File     │
│  Detection  │ │  Module   │ │  Client  │ │  Manager   │
├─────────────┤ ├───────────┤ ├──────────┤ ├────────────┤
│ Screenshot  │ │  Mouse    │ │  HTTP    │ │ Directory  │
│ ORB / TM    │ │ Keyboard  │ │  Retry   │ │ Path Gen   │
│ Processor   │ │ Window    │ │  Parse   │ │ Verify     │
└─────────────┘ └───────────┘ └──────────┘ └────────────┘
```

## 📋 Requirements

- **OS**: Windows 10 or 11
- **Python**: 3.10+
- **Desktop**: Notepad shortcut should be visible on the desktop (for initial launch)

## 🚀 Quick Start

### 1. Install uv (Recommended)

```powershell
# Using pip
pip install uv

# Or using PowerShell installer
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup

```powershell
cd notepad-automation

# Create virtual environment and install dependencies
uv sync

# Install the package in development mode
uv pip install -e .
```

### 3. Run the Automation

```powershell
# Using the entry point
uv run notepad-auto

# Or directly
uv run python -m notepad_automation.main
```

## 📁 Project Structure

```
notepad-automation/
├── pyproject.toml                    # Project configuration
├── README.md                         # This file
├── src/
│   └── notepad_automation/
│       ├── __init__.py
│       ├── main.py                   # Entry point & orchestrator
│       ├── config.py                 # Configuration constants
│       ├── find_icon_coordinates/
│       │   └── find_icon_coordinates.py # ORB & Template Matching logic
│       ├── automation/
│       │   ├── mouse.py              # Mouse control
│       │   ├── keyboard.py           # Keyboard automation
│       │   └── window.py             # Window validation & Win32 API
│       ├── api/
│       │   └── client.py             # JSONPlaceholder client
│       ├── files/
│       │   └── manager.py            # File operations
│       └── supported_icons/          # Prioritized icon templates (1.png, 2.png, etc.)
├── tests/
│   └── ...                           # Test files
├── screenshots/                      # Runtime screenshots for debugging
└── logs/                             # Runtime logs
```

## 🔬 How Vision Grounding Works

### Primary: ORB Feature Matching
ORB (Oriented FAST and Rotated BRIEF) is a fast and robust local feature detector and descriptor.
1. **Feature Extraction**: Detects keypoints in both the reference icon and the current desktop screenshot.
2. **Descriptor Matching**: Matches descriptors using Hamming distance.
3. **RANSAC Homography**: Estimates the transformation matrix to find the icon's center even if it's slightly distorted or partially obscured.

### Secondary: Multi-Scale Template Matching
If ORB fails (common with very simple or low-texture icons), the system uses multi-scale template matching:
1. **Pyramid Scaling**: Resizes the reference icon across multiple scales (0.25x to 2.0x).
2. **Normalized Cross-Correlation**: Finds the best match in the screenshot for each scale.
3. **Thresholding**: Selects the match with the highest confidence score above a configurable threshold.

## 🛡️ Error Handling

- **Retry Logic**: Retries icon detection and API calls with configurable delays.
- **Safe Window Targeting**: Uses Win32 API to verify PID and process names before interacting, preventing accidental interaction with the wrong windows.
- **Graceful Shutdown**: Handles interrupts and failures by attempting to clean up opened windows.

## 📊 Output

- **Files**: Created in `~/Desktop/tjm-project/` by default.
- **Logs**: Detailed execution logs in `logs/automation.log`.
- **Debug Media**: 
    - Original screenshots: `screenshots/desktop_current.png`
    - Annotated detections: `screenshots/annotated/annotated_at_YYYY-MM-DD_HH-MM-SS.png` (highlights the matched icon and template name)

## 📚 Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python`| Core CV operations (ORB, Template Matching) |
| `numpy` | Numerical operations on image arrays |
| `pyautogui` | Cross-platform mouse and keyboard control |
| `pywin32` | Windows-specific API access for window management |
| `httpx` | Modern, async-capable HTTP client for API interaction |
| `psutil` | Process management and verification |
| `pytesseract`| OCR capabilities (optional/extended features) |

## 📄 License

MIT License - feel free to use and modify.
