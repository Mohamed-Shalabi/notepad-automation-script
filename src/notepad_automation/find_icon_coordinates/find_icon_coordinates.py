import logging
import cv2
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def preprocess_icon(icon: np.ndarray, min_size: int = 64) -> np.ndarray:
    """
    Preprocess icon for ORB:
    - Remove alpha channel if present
    - Convert to grayscale
    - Resize small icons to at least min_size
    - Optional contrast enhancement
    """
    # Remove alpha if exists
    if icon.ndim == 3 and icon.shape[2] == 4:
        icon = cv2.cvtColor(icon, cv2.COLOR_BGRA2BGR)
    # Convert to grayscale
    if icon.ndim == 3:
        icon = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    # Resize if too small
    h, w = icon.shape
    scale = max(min_size / h, min_size / w, 1.0)
    if scale > 1.0:
        icon = cv2.resize(icon, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # Enhance contrast
    icon = cv2.equalizeHist(icon)
    return icon

def load_grayscale(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image

def compute_orb_features(
    image: np.ndarray,
    nfeatures: int = 1500
):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(image, None)
    logger.debug(f"ORB: Detected {len(kp)} keypoints")
    return kp, des

def estimate_center_from_matches(
    kp_icon, kp_screen, matches, icon_shape
) -> Optional[Tuple[int, int]]:
    src_pts = np.float32(
        [kp_icon[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp_screen[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        logger.debug("ORB: Failed to find homography matrix")
        return None

    h, w = icon_shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H)

    center = (
        int(projected[:, 0, 0].mean()),
        int(projected[:, 0, 1].mean())
    )
    logger.debug(f"ORB: Estimated center at {center}")
    return center

def find_with_orb(
    screenshot: np.ndarray,
    icon: np.ndarray,
    min_matches: int = 12
) -> Optional[Tuple[int, int]]:
    kp_i, des_i = compute_orb_features(icon, nfeatures=2000)
    logger.debug(f"ORB: Icon Features - Keypoints: {len(kp_i)}, Descriptors shape: {des_i.shape if des_i is not None else 'None'}")
    if des_i is None:
        for scale in np.linspace(2, 0.25, 10):
            resizedIcon = cv2.resize(icon, None, fx=scale, fy=scale)
            kp_i, des_i = compute_orb_features(resizedIcon)
            if des_i is not None:
                break


    kp_s, des_s = compute_orb_features(screenshot)
    logger.debug(f"ORB: Screenshot Features - Keypoints: {len(kp_s)}, Descriptors shape: {des_s.shape if des_s is not None else 'None'}")

    if des_i is None or des_s is None:
        return None

    logger.debug("ORB: Matching features...")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(des_i, des_s, k=2)

    good = [
        m for m, n in raw_matches
        if m.distance < 0.75 * n.distance
    ]
    logger.debug(f"ORB: Found {len(good)} good matches (threshold: {min_matches})")

    if len(good) < min_matches:
        logger.debug("ORB: Too few good matches found" + str(len(good)) + " < " + str(min_matches))
        return None

    return estimate_center_from_matches(
        kp_i, kp_s, good, icon.shape
    )

def find_with_template_matching(
    screenshot: np.ndarray,
    icon: np.ndarray,
    threshold: float = 0.75
) -> Optional[Tuple[int, int]]:
    best_score, best_center = 0, None
    best_scale = 1.0

    logger.debug(f"Template Matching: Starting multi-scale search (threshold: {threshold})")
    for scale in np.linspace(0.25, 2, 40):
        resized = cv2.resize(icon, None, fx=scale, fy=scale)
        if resized.shape[0] > screenshot.shape[0] or resized.shape[1] > screenshot.shape[1]:
            logger.debug(f"Template Matching: Skipping scale {scale:.2f} (resized icon larger than screenshot)")
            continue

        result = cv2.matchTemplate(
            screenshot, resized, cv2.TM_CCOEFF_NORMED
        )

        _, score, _, loc = cv2.minMaxLoc(result)
        if score > best_score:
            h, w = resized.shape
            best_score = score
            best_center = (loc[0] + w // 2, loc[1] + h // 2)
            best_scale = scale
            logger.debug(f"Template Matching: Found better match at scale {scale:.2f} with score {score:.4f}")

    if best_score < threshold:
        logger.debug(f"Template Matching: Failed. Best score {best_score:.4f} was below threshold {threshold}")
    else:
        logger.debug(f"Template Matching: Success! Best score {best_score:.4f} at scale {best_scale:.2f}")

    return best_center if best_score >= threshold else None

def find_icon_coordinates(
    screenshot_path: str,
    icon_path: str
) -> Optional[Tuple[int, int]]:
    screenshot = load_grayscale(screenshot_path)
    icon = preprocess_icon(cv2.imread(icon_path, cv2.IMREAD_UNCHANGED))


    logger.info(f"Attempting icon detection using ORB feature matching...")
    result = find_with_orb(screenshot, icon)
    if result is not None:
        logger.info(f"Icon detected using ORB at {result}")
        return result

    logger.info("ORB detection failed. Attempting Template Matching with scaling...")
    result = find_with_template_matching(screenshot, icon)
    if result is not None:
        logger.info(f"Icon detected using Template Matching at {result}")
    else:
        logger.warning("Icon detection failed: No matches found with ORB or Template Matching.")
    
    return result
