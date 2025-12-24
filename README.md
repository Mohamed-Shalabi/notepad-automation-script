# Notepad Automation System

A **production-ready Windows desktop automation system** that uses **CLIP-based vision grounding** to detect and interact with the Notepad desktop icon.

## 🎯 Overview

This system demonstrates **semantic vision grounding** - a technique that uses vision-language models to find UI elements based on their meaning, not fixed coordinates or exact pixel matching.

### Key Features

- **Vision-Based Grounding**: Uses OpenAI's CLIP model to find the Notepad icon by understanding "what is a Notepad icon" rather than pixel matching
- **Robust Detection**: Works regardless of icon position, size, theme, or background
- **Full Automation**: Fetches data from API, creates files, saves them, repeats for 10 posts
- **Comprehensive Error Handling**: Retries, timeouts, graceful degradation
- **Detailed Logging**: Complete audit trail of all actions

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
│   Vision    │ │Automation │ │   API    │ │   File     │
│  Grounding  │ │  Module   │ │  Client  │ │  Manager   │
├─────────────┤ ├───────────┤ ├──────────┤ ├────────────┤
│ Screenshot  │ │  Mouse    │ │  HTTP    │ │ Directory  │
│ CLIP Model  │ │ Keyboard  │ │  Retry   │ │ Path Gen   │
│ Detector    │ │ Window    │ │  Parse   │ │ Verify     │
└─────────────┘ └───────────┘ └──────────┘ └────────────┘
```

## 📋 Requirements

- **OS**: Windows 10 or 11
- **Resolution**: 1920×1080 (primary monitor)
- **Python**: 3.10+
- **Desktop**: Notepad shortcut must exist on desktop

## 🚀 Quick Start

### 1. Install uv (if not already installed)

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
│       ├── grounding/
│       │   ├── __init__.py
│       │   ├── screenshot.py         # Desktop capture
│       │   └── detector.py           # CLIP-based detection
│       ├── automation/
│       │   ├── __init__.py
│       │   ├── mouse.py              # Mouse control
│       │   ├── keyboard.py           # Keyboard automation
│       │   └── window.py             # Window validation
│       ├── api/
│       │   ├── __init__.py
│       │   └── client.py             # JSONPlaceholder client
│       └── files/
│           ├── __init__.py
│           └── manager.py            # File operations
├── tests/
│   └── ...                           # Test files
├── screenshots/                      # Annotated detection screenshots
└── logs/                             # Runtime logs
```

## 🔬 How Vision Grounding Works

### Traditional Template Matching (❌ Not Used)
```
Screenshot → Find exact pixel pattern → Return coordinates
```
- Requires pre-stored reference images
- Breaks with different themes, sizes, positions
- No semantic understanding

### CLIP-Based Semantic Grounding (✅ Our Approach)
```
Screenshot → Extract regions → CLIP encode each region
Text prompts → CLIP encode ("a Notepad icon")
Compare embeddings → Rank by similarity → Return best match
```

### Why CLIP?

**CLIP (Contrastive Language-Image Pre-training)** understands the *meaning* of images:

1. **Trained on 400M image-text pairs** from the internet
2. **Maps images and text to the same embedding space**
3. **Can compare arbitrary images to text descriptions**

When we ask "which region looks like a Notepad icon?", CLIP understands:
- What text editors generally look like
- The concept of "notepad" or "document editing"
- Visual patterns associated with such icons

### Detection Pipeline

```python
# 1. Capture fresh screenshot
screenshot = capture_desktop_screenshot()

# 2. Extract candidate regions (sliding window)
candidates = extract_candidates(screenshot, sizes=[48, 64, 80, 96])

# 3. Encode with CLIP vision encoder
image_embeddings = clip.encode_images(candidates)

# 4. Encode text prompts
text_prompts = [
    "a Notepad application icon",
    "a text editor icon",
    "a Windows Notepad shortcut"
]
text_embeddings = clip.encode_text(text_prompts)

# 5. Find best match
similarities = image_embeddings @ text_embeddings.T
best_candidate = candidates[similarities.argmax()]

# 6. Return center coordinates
return best_candidate.center_x, best_candidate.center_y
```

## 🛡️ Error Handling

| Scenario | Handling |
|----------|----------|
| Icon not found | Retry 3 times with 1s delay |
| Multiple matches | Rank by confidence, select highest |
| API unavailable | Retry with exponential backoff, then abort |
| Notepad won't launch | Retry 2 times, then skip post |
| File save fails | Verify file exists, report error |
| Unexpected popup | Check window title, close if needed |

All failures are:
- **Logged** with full context
- **Non-crashing** (graceful degradation)
- **Reported** in final summary

## 📊 Output

### Files Created
```
Desktop/
└── tjm-project/
    ├── post_1.txt
    ├── post_2.txt
    ├── post_3.txt
    └── ... (10 files total)
```

### File Format
```
Title: sunt aut facere repellat provident occaecati excepturi optio reprehenderit

quia et suscipit
suscipit recusandae consequuntur expedita et cum
reprehenderit molestiae ut ut quas totam
nostrum rerum est autem sunt rem eveniet architecto
```

### Screenshots
```
screenshots/
├── detection_step_0.png        # Detection for first post
├── detection_step_4.png        # Detection for middle post
└── detection_step_9.png        # Detection for last post
```

### Logs
```
logs/
└── automation.log              # Complete execution log
```

## 🎤 Interview Discussion Points

### Why CLIP over Template Matching?

1. **Generalization**: Works on any icon appearance without reference images
2. **Semantic Understanding**: Knows what a "text editor" looks like conceptually
3. **Robustness**: Handles theme changes, scaling, partial visibility
4. **Extensibility**: Same approach works for any UI element

### Known Limitations

1. **Speed**: Sliding window + CLIP is slower than direct detection
2. **Very Small Icons**: May not be detected at extreme scales
3. **Ambiguous Icons**: Similar apps (Notepad vs Notepad++) may confuse detector
4. **GPU Recommended**: CPU inference is 5-10x slower

### Potential Improvements

1. **YOLO + CLIP Hybrid**: Use fast object detector for proposals, CLIP for ranking
2. **Fine-tuning**: Train on desktop icon dataset for better accuracy
3. **Caching**: Store embeddings of known desktop regions
4. **Multi-monitor**: Extend to all connected displays

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Screenshot Capture | ~50ms |
| Candidate Extraction | ~200ms |
| CLIP Inference (GPU) | ~100ms per batch |
| CLIP Inference (CPU) | ~1s per batch |
| Total Detection | 3-5s |

## 📚 Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `transformers` | CLIP model loading |
| `Pillow` | Image processing |
| `mss` | Fast screenshots |
| `pyautogui` | Mouse/keyboard control |
| `pywin32` | Windows API access |
| `httpx` | HTTP client |
| `opencv-python` | Image operations |

## 📄 License

MIT License - feel free to use and modify.

## 🙏 Acknowledgments

- OpenAI for the CLIP model
- DummyJSON for the test API
- The Python automation community
