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
        Detect the Notepad icon in a screenshot.
        
        Args:
            screenshot: The desktop screenshot to search.
            custom_prompts: Optional custom text prompts (uses config defaults if None).
        
        Returns:
            DetectionResult with the best match and all candidates.
        """
        self.load_model()
        
        prompts = custom_prompts or config.grounding.notepad_prompts
        logger.info(f"Detecting Notepad icon with {len(prompts)} prompts")
        
        # Extract candidate regions using sliding window
        candidates = self._extract_candidates(screenshot)
        logger.info(f"Extracted {len(candidates)} candidate regions")
        
        if not candidates:
            return DetectionResult(
                found=False,
                center_x=0, center_y=0,
                bbox_x=0, bbox_y=0,
                bbox_width=0, bbox_height=0,
                confidence=0.0,
                all_candidates=[]
            )
        
        # Prepare full image tensor for efficient scoring
        # CLIP expects normalized tensors in range [0, 1] then normalized with mean/std
        # TF.to_tensor handles the [0, 1] scaling
        img_tensor = TF.to_tensor(screenshot.image).to(self.device).float()
        
        # Get normalization parameters from processor
        # Use standard CLIP values if not found (typically these)
        mean_vals = getattr(self.processor.image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        std_vals = getattr(self.processor.image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        
        mean = torch.tensor(mean_vals).view(3, 1, 1).to(self.device)
        std = torch.tensor(std_vals).view(3, 1, 1).to(self.device)
        img_tensor = (img_tensor - mean) / std
        
        # Score each candidate against the prompts using optimized batched tensor crops
        scored_candidates = self._score_candidates(img_tensor, candidates, prompts)
        
        # Apply non-maximum suppression
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
        
        # Get the best candidate
        best = filtered_candidates[0]
        
        # Check confidence threshold
        if best["confidence"] < config.grounding.confidence_threshold:
            logger.warning(
                f"Best candidate confidence {best['confidence']:.3f} "
                f"below threshold {config.grounding.confidence_threshold}"
            )
            return DetectionResult(
                found=False,
                center_x=best["center_x"],
                center_y=best["center_y"],
                bbox_x=best["x"],
                bbox_y=best["y"],
                bbox_width=best["width"],
                bbox_height=best["height"],
                confidence=best["confidence"],
                all_candidates=filtered_candidates
            )
        
        logger.info(
            f"Notepad icon detected at ({best['center_x']}, {best['center_y']}) "
            f"with confidence {best['confidence']:.3f}"
        )
        
        return DetectionResult(
            found=True,
            center_x=best["center_x"],
            center_y=best["center_y"],
            bbox_x=best["x"],
            bbox_y=best["y"],
            bbox_width=best["width"],
            bbox_height=best["height"],
            confidence=best["confidence"],
            all_candidates=filtered_candidates
        )
    
    def _extract_candidates(
        self,
        screenshot: ScreenshotResult
    ) -> List[dict]:
        """
        Extract candidate regions using optimized grid-based approach.
        
        Desktop icons are typically:
        - Spread across the screen or in a grid pattern
        - In a grid pattern (~80px apart)
        - Between 32-96px in size
        
        This optimized version focuses on grid-based extraction to reduce
        the number of candidates while covering the full screen.
        """
        candidates = []
        image = screenshot.image
        img_width, img_height = image.size
        
        # Scan the full screen width
        scan_width = img_width
        
        # Use a grid pattern from config
        grid_step = config.grounding.window_stride
        
        # Use window sizes from config
        window_sizes = config.grounding.window_sizes
        
        for window_size in window_sizes:
            # Scan the screen
            for y in range(20, img_height - window_size - 20, grid_step):
                for x in range(20, scan_width, grid_step):
                    candidates.append({
                        "x": x,
                        "y": y,
                        "width": window_size,
                        "height": window_size,
                        "center_x": x + window_size // 2,
                        "center_y": y + window_size // 2
                    })
            

        
        return candidates
    
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
            max_scores = similarities.max(axis=1)
            
            for j, candidate in enumerate(batch):
                candidate_copy = candidate.copy()
                candidate_copy["confidence"] = float(max_scores[j])
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
