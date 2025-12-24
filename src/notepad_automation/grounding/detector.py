"""
CLIP-Based Icon Detector

This module implements semantic icon detection using OpenAI's CLIP model.
Instead of template matching, it uses natural language understanding to
find icons based on their semantic meaning.

Key Advantages:
- No pre-stored reference images needed
- Works across different icon themes and sizes
- Generalizes to icons never seen before
- Robust to visual variations

Algorithm:
1. Extract candidate regions using sliding window
2. Encode each region with CLIP vision encoder
3. Encode text prompts describing the target icon
4. Compute similarity between image and text embeddings
5. Apply non-maximum suppression to remove duplicates
6. Return best candidate with confidence score
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from ..config import config
from .screenshot import ScreenshotResult


logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of an icon detection operation."""
    found: bool
    center_x: int
    center_y: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    confidence: float
    all_candidates: List[dict]


class IconDetector:
    """
    CLIP-based semantic icon detector.
    
    Uses a vision-language model to find icons on the desktop by
    matching image regions to text descriptions of the target icon.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the icon detector.
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing IconDetector on device: {self.device}")
        
        self.model = None
        self.processor = None
        self._model_loaded = False
    
    def load_model(self) -> None:
        """Load the CLIP model and processor."""
        if self._model_loaded:
            return
        
        logger.info(f"Loading CLIP model: {config.grounding.clip_model_name}")
        
        try:
            self.processor = CLIPProcessor.from_pretrained(
                config.grounding.clip_model_name,
                use_fast=True
            )
            self.model = CLIPModel.from_pretrained(
                config.grounding.clip_model_name
            ).to(self.device)
            self.model.eval()
            
            self._model_loaded = True
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"CLIP model loading failed: {e}") from e
    
    def detect_notepad_icon(
        self,
        screenshot: ScreenshotResult,
        custom_prompts: Optional[List[str]] = None
    ) -> DetectionResult:
        """
        Detect the Notepad icon in a screenshot with resolution-adaptive scaling.
        
        Args:
            screenshot: The desktop screenshot to search.
            custom_prompts: Optional custom text prompts.
        
        Returns:
            DetectionResult with coordinates mapped to the original resolution.
        """
        self.load_model()
        
        # 1. Get scaled dimensions for detection (targets: 990, 1280, 1920, 2560)
        orig_w, orig_h = screenshot.image.size
        target_w, target_h, scale, dyn_windows, dyn_stride = self._get_scaled_dimensions(orig_w, orig_h)
        
        # Prepare the image for detection (scale down if necessary)
        if scale < 1.0:
            logger.info(f"Downscaling screenshot from {orig_w}x{orig_h} to {target_w}x{target_h} (scale: {scale:.2f})")
            detect_image = screenshot.image.resize((target_w, target_h), Image.LANCZOS)
        else:
            detect_image = screenshot.image

        prompts = custom_prompts or config.grounding.notepad_prompts
        
        logger.info(f"Detecting Notepad icon with {len(prompts)} prompts")
        
        # 2. Extract candidate regions from the (possibly scaled) image
        # Mocking a ScreenshotResult-like object for _extract_candidates
        # since it only uses the .image attribute currently.
        class ScaledResult:
            def __init__(self, img): self.image = img
            
        # Use dynamic parameters calculated by scaling logic above
        
        candidates = self._extract_candidates(
            ScaledResult(detect_image),
            window_sizes=dyn_windows,
            window_stride=dyn_stride
        )
        logger.info(f"Extracted {len(candidates)} candidate regions using stride {dyn_stride or config.grounding.window_stride}")
        
        if not candidates:
            return DetectionResult(
                found=False,
                center_x=0, center_y=0,
                bbox_x=0, bbox_y=0,
                bbox_width=0, bbox_height=0,
                confidence=0.0,
                all_candidates=[]
            )
        
        # 3. Prepare full image tensor for efficient scoring
        img_tensor = TF.to_tensor(detect_image).to(self.device).float()
        
        # Normalize
        mean_vals = getattr(self.processor.image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        std_vals = getattr(self.processor.image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        mean = torch.tensor(mean_vals).view(3, 1, 1).to(self.device)
        std = torch.tensor(std_vals).view(3, 1, 1).to(self.device)
        img_tensor = (img_tensor - mean) / std
        
        # Score candidates
        scored_candidates = self._score_candidates(img_tensor, candidates, prompts)
        
        # 4. Filter and Remap
        filtered_candidates = self._apply_nms(scored_candidates)
        
        if not filtered_candidates:
            return DetectionResult(
                found=False,
                center_x=0, center_y=0,
                bbox_x=0, bbox_y=0,
                bbox_width=0, bbox_height=0,
                confidence=0.0,
                all_candidates=[]
            )
        
        # Get the best candidate and map back to original resolution
        best = filtered_candidates[0]
        
        # Simple lambda to rescale coordinates
        rescale = lambda x: int(x / scale)
        
        remapped_best = {
            "center_x": rescale(best["center_x"]),
            "center_y": rescale(best["center_y"]),
            "x": rescale(best["x"]),
            "y": rescale(best["y"]),
            "width": rescale(best["width"]),
            "height": rescale(best["height"]),
            "confidence": best["confidence"]
        }
        
        # Check confidence threshold
        if remapped_best["confidence"] < config.grounding.confidence_threshold:
            logger.warning(
                f"Best candidate confidence {remapped_best['confidence']:.3f} "
                f"below threshold {config.grounding.confidence_threshold}"
            )
            return DetectionResult(
                found=False,
                center_x=remapped_best["center_x"],
                center_y=remapped_best["center_y"],
                bbox_x=remapped_best["x"],
                bbox_y=remapped_best["y"],
                bbox_width=remapped_best["width"],
                bbox_height=remapped_best["height"],
                confidence=remapped_best["confidence"],
                all_candidates=filtered_candidates # Keeping these in scaled space for overlay if needed, or remap?
            )
            # Note: annotations usually happen on scaled space if we use the saved scaled screenshot, 
            # but main.py saves the original screenshot usually. 
            # For simplicity, we keep all_candidates in scaled space for now, 
            # but we return the remapped primary detection.
        
        logger.info(
            f"Notepad icon detected at ({remapped_best['center_x']}, {remapped_best['center_y']}) "
            f"(original resolution) with confidence {remapped_best['confidence']:.3f}"
        )
        
        return DetectionResult(
            found=True,
            center_x=remapped_best["center_x"],
            center_y=remapped_best["center_y"],
            bbox_x=remapped_best["x"],
            bbox_y=remapped_best["y"],
            bbox_width=remapped_best["width"],
            bbox_height=remapped_best["height"],
            confidence=remapped_best["confidence"],
            all_candidates=filtered_candidates
        )
    
    def _extract_candidates(
        self,
        screenshot: ScreenshotResult,
        window_sizes: Optional[List[int]] = None,
        window_stride: Optional[int] = None
    ) -> List[dict]:
        """
        Extract candidate regions using optimized grid-based approach.
        
        Args:
            screenshot: The screenshot result.
            window_sizes: Optional override for window sizes.
            window_stride: Optional override for grid step.
        """
        candidates = []
        image = screenshot.image
        img_width, img_height = image.size
        
        # Use dynamic parameters or fall back to config
        grid_step = window_stride or config.grounding.window_stride
        sizes = window_sizes or config.grounding.window_sizes
        
        for window_size in sizes:
            # Scan the screen
            # Start at a small margin (20px) to avoid taskbar/edges if possible
            for y in range(20, img_height - window_size - 20, grid_step):
                for x in range(20, img_width - window_size - 20, grid_step):
                    candidates.append({
                        "x": x,
                        "y": y,
                        "width": window_size,
                        "height": window_size,
                        "center_x": x + window_size // 2,
                        "center_y": y + window_size // 2
                    })
            

        
        return candidates
    
    def _get_scaled_dimensions(self, width: int, height: int) -> Tuple[int, int, float, Optional[List[int]], Optional[int]]:
        """
        Determine target dimensions and dynamic window parameters.
        
        Tiers (width >= threshold -> target_width):
        - 4032 (4K+5%) -> 2560 (Windows: [80, 130, 180], Stride: 40)
        - 3840 (4K)    -> 1920 (Windows: [64, 95, 130], Stride: 32)
        - 2560 (2K)    -> 1280 (Windows: [40, 65, 90], Stride: 20)
        - 1920 (FHD)   -> 990  (Windows: [32, 50, 70], Stride: 16)

        These tiers are calculated for 9:16 screens to notepad app in all scale values from 100% to 175%
        """
        # Define tiers: (threshold, target_w, window_sizes, stride)
        tiermaps = [
            (4032, 2560, [100, 150], 93),
            (3840, 1920, [75, 110], 70),
            (2560, 1280, [54, 78], 48),
            (1920, 990, [42, 60], 36)
        ]
        
        target_w = width
        dyn_windows = None
        dyn_stride = None
        
        for threshold, target, windows, stride in tiermaps:
            if width >= threshold:
                target_w = target
                dyn_windows = windows
                dyn_stride = stride
                break
        
        if target_w == width:
            return width, height, 1.0, None, None
            
        scale_factor = target_w / width
        target_h = int(height * scale_factor)
        return target_w, target_h, scale_factor, dyn_windows, dyn_stride
    
    def _score_candidates(
        self,
        full_image_tensor: torch.Tensor,
        candidates: List[dict],
        prompts: List[str],
        batch_size: int = 128
    ) -> List[dict]:
        """
        Score each candidate region using efficient batched tensor operations.
        
        Args:
            full_image_tensor: Normalized full screenshot tensor (C, H, W)
            candidates: List of region coordinates
            prompts: Text descriptions
            batch_size: Number of regions to process in parallel
            
        Returns:
            List of candidates with confidence scores.
        """
        scored = []
        
        # 1. Pre-compute text features
        with torch.no_grad():
            text_inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 2. Process image regions in batches
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            
            # Efficiently extract and resize regions using torch
            region_tensors = []
            for c in batch:
                # TF.crop is efficient on tensors
                region = TF.crop(
                    full_image_tensor, 
                    c["y"], c["x"], 
                    c["height"], c["width"]
                )
                # Resize to CLIP's expected input size (224x224)
                region = F.interpolate(
                    region.unsqueeze(0), 
                    size=(224, 224), 
                    mode="bilinear", 
                    align_corners=False
                ).squeeze(0)
                region_tensors.append(region)
            
            # Stack into a single batch tensor
            batch_tensor = torch.stack(region_tensors).to(self.device)
            
            # 3. Compute image features and similarities
            with torch.no_grad():
                image_features = self.model.get_image_features(pixel_values=batch_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Batch matrix multiplication for similarities
                # (batch_size, embed_dim) @ (embed_dim, num_prompts)
                similarities = (image_features @ text_features.T).cpu().numpy()
            
            # Take max similarity across all prompts
            batch_scores = similarities.max(axis=1)
            
            for j, candidate in enumerate(batch):
                candidate_copy = candidate.copy()
                candidate_copy["confidence"] = float(batch_scores[j])
                scored.append(candidate_copy)
                
        return sorted(scored, key=lambda x: x["confidence"], reverse=True)
    
    def _apply_nms(
        self,
        candidates: List[dict],
        iou_threshold: Optional[float] = None
    ) -> List[dict]:
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Keeps the highest confidence detection in each region.
        """
        if not candidates:
            return []
        
        iou_threshold = iou_threshold or config.grounding.nms_iou_threshold
        
        # Already sorted by confidence
        kept = []
        
        for candidate in candidates:
            # Check if this candidate overlaps with any kept candidate
            should_keep = True
            
            for kept_candidate in kept:
                iou = self._compute_iou(candidate, kept_candidate)
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept.append(candidate)
        
        return kept
    
    @staticmethod
    def _compute_iou(box1: dict, box2: dict) -> float:
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_with_retry(
        self,
        screenshot_func,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ) -> DetectionResult:
        """
        Attempt detection with retries on failure.
        
        Args:
            screenshot_func: Function that returns a ScreenshotResult.
            max_retries: Maximum retry attempts (uses config default if None).
            retry_delay: Delay between retries in seconds.
        
        Returns:
            DetectionResult from the successful attempt or the last attempt.
        """
        import time
        
        max_retries = max_retries or config.grounding.max_retries
        retry_delay = retry_delay or config.grounding.retry_delay_seconds
        
        for attempt in range(max_retries):
            logger.info(f"Detection attempt {attempt + 1}/{max_retries}")
            
            try:
                screenshot = screenshot_func()
                result = self.detect_notepad_icon(screenshot)
                
                if result.found:
                    logger.info(f"Detection succeeded on attempt {attempt + 1}")
                    return result
                
                logger.warning(f"Detection failed on attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Detection error on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Waiting {retry_delay}s before retry...")
                time.sleep(retry_delay)
        
        logger.error(f"All {max_retries} detection attempts failed")
        return DetectionResult(
            found=False,
            center_x=0, center_y=0,
            bbox_x=0, bbox_y=0,
            bbox_width=0, bbox_height=0,
            confidence=0.0,
            all_candidates=[]
        )
