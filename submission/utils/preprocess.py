"""
Image preprocessing utilities for enhanced VLM accuracy
"""

import base64
import cv2
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess image for better VLM text detection
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Failed to load image with cv2, using PIL")
            img = np.array(Image.open(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        logger.info(f"Original image shape: {img.shape}")
        
        # 1. Resize if too large or too small (optimal for VLM: 1500-2000px width)
        height, width = img.shape[:2]
        target_width = 1800
        if width > 2500 or width < 800:
            scale = target_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            logger.info(f"Resized to: {img.shape}")
        
        # 2. Denoise while preserving text edges
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # 3. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.merge([l, a, b])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        
        # 4. Sharpen text
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        
        # 5. Auto-adjust brightness and contrast
        img = auto_adjust_brightness_contrast(img)
        
        logger.info("Image preprocessing completed successfully")
        return img
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}, returning original")
        # Return original if preprocessing fails
        return cv2.imread(image_path)


def auto_adjust_brightness_contrast(img: np.ndarray, clip_hist_percent: float = 1.0) -> np.ndarray:
    """Automatically adjust brightness and contrast
    
    Args:
        img: Input image
        clip_hist_percent: Percentage to clip histogram
        
    Returns:
        Adjusted image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0
    
    # Find minimum
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Find maximum
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    # Apply adjustment
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def encode_image(image_path: str, preprocess: bool = True) -> str:
    """Encode image to base64 with optional preprocessing
    
    Args:
        image_path: Path to image file
        preprocess: Whether to apply preprocessing
        
    Returns:
        Base64 encoded string
    """
    try:
        if preprocess:
            # Preprocess image first
            img = preprocess_image(image_path)
            
            # Convert to PIL Image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Encode to base64
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG", optimize=True)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            # Original behavior - no preprocessing
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
                
    except Exception as e:
        logger.error(f"Image encoding failed: {e}, using original")
        # Fallback to original
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
