"""
OCR Module - Multi-pass OCR with consensus for robust text extraction
Uses EasyOCR and Tesseract for redundancy and confidence scoring
"""

import cv2
import numpy as np
import easyocr
import pytesseract
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiPassOCR:
    """
    Multi-pass OCR system using EasyOCR and Tesseract
    Implements consensus mechanism to improve extraction accuracy without ground truth
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize OCR engines
        
        Args:
            use_gpu: Whether to use GPU acceleration for EasyOCR
        """
        self.use_gpu = use_gpu
        logger.info("Initializing EasyOCR...")
        self.easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
        logger.info("OCR engines initialized")
    
    def preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Create multiple preprocessed versions for multi-pass OCR
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of preprocessed image variants
        """
        variants = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Variant 1: Original grayscale
        variants.append(gray)
        
        # Variant 2: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        variants.append(adaptive)
        
        # Variant 3: Denoised + sharpened
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        variants.append(sharpened)
        
        # Variant 4: High contrast OTSU
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        
        return variants
    
    def extract_text_easyocr(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text using EasyOCR
        
        Args:
            image: Preprocessed image
        
        Returns:
            List of detection dictionaries with text, bbox, and confidence
        """
        results = self.easyocr_reader.readtext(image)
        
        detections = []
        for bbox, text, conf in results:
            detections.append({
                'text': text,
                'bbox': bbox,
                'confidence': conf,
                'source': 'easyocr'
            })
        
        return detections
    
    def extract_text_tesseract(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text using Tesseract OCR
        
        Args:
            image: Preprocessed image
        
        Returns:
            List of detection dictionaries with text, bbox, and confidence
        """
        # Get detailed data from Tesseract
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        detections = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # Filter low confidence and empty text
            if conf > 30 and text:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                
                detections.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': conf / 100.0,  # Normalize to 0-1
                    'source': 'tesseract'
                })
        
        return detections
    
    def consensus_merge(self, detections_list: List[List[Dict]]) -> List[Dict]:
        """
        Merge detections from multiple OCR passes using spatial and textual consensus
        
        Args:
            detections_list: List of detection lists from different passes
        
        Returns:
            Merged list of high-confidence detections
        """
        # Flatten all detections
        all_detections = []
        for detections in detections_list:
            all_detections.extend(detections)
        
        if not all_detections:
            return []
        
        # Group spatially close detections
        merged = []
        used = set()
        
        for i, det in enumerate(all_detections):
            if i in used:
                continue
            
            # Find overlapping/similar detections
            similar_group = [det]
            used.add(i)
            
            for j, other_det in enumerate(all_detections):
                if j <= i or j in used:
                    continue
                
                # Check if bboxes overlap or text matches
                if self._bbox_overlap(det['bbox'], other_det['bbox']) > 0.5 or \
                   self._text_similarity(det['text'], other_det['text']) > 0.8:
                    similar_group.append(other_det)
                    used.add(j)
            
            # Aggregate the group
            if similar_group:
                merged_det = self._aggregate_detections(similar_group)
                merged.append(merged_det)
        
        return merged
    
    def _bbox_overlap(self, bbox1: List, bbox2: List) -> float:
        """Calculate IoU between two bboxes"""
        # Convert to x1, y1, x2, y2
        def bbox_to_coords(bbox):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            return min(xs), min(ys), max(xs), max(ys)
        
        x1_min, y1_min, x1_max, y1_max = bbox_to_coords(bbox1)
        x2_min, y2_min, x2_max, y2_max = bbox_to_coords(bbox2)
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _aggregate_detections(self, group: List[Dict]) -> Dict:
        """Aggregate multiple detections of the same region"""
        # Use highest confidence text
        group_sorted = sorted(group, key=lambda x: x['confidence'], reverse=True)
        best_detection = group_sorted[0]
        
        # Average confidence across sources
        avg_confidence = sum(d['confidence'] for d in group) / len(group)
        
        # Prefer EasyOCR text if available
        easyocr_texts = [d['text'] for d in group if d['source'] == 'easyocr']
        final_text = easyocr_texts[0] if easyocr_texts else best_detection['text']
        
        return {
            'text': final_text,
            'bbox': best_detection['bbox'],
            'confidence': avg_confidence,
            'consensus_count': len(group)
        }
    
    def extract_with_layout(self, image: np.ndarray) -> Dict:
        """
        Extract text with layout information (spatial structure)
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with full_text, detections, and layout structure
        """
        # Preprocess image variants
        variants = self.preprocess_image(image)
        
        # Run multi-pass OCR
        all_detections = []
        
        # Pass 1: EasyOCR on best variant (original grayscale)
        easyocr_det = self.extract_text_easyocr(variants[0])
        all_detections.append(easyocr_det)
        
        # Pass 2: Tesseract on adaptive threshold (skip if not installed)
        try:
            tesseract_det = self.extract_text_tesseract(variants[1])
            all_detections.append(tesseract_det)
        except Exception as e:
            logger.warning(f"Tesseract not available, using EasyOCR only: {e}")
        
        # Consensus merge
        merged_detections = self.consensus_merge(all_detections)
        
        # Sort by vertical position (top to bottom)
        merged_detections.sort(key=lambda d: min(p[1] for p in d['bbox']))
        
        # Build full text
        full_text = " ".join([d['text'] for d in merged_detections])
        
        # Detect layout regions (header, body, footer)
        layout = self._detect_layout_regions(image, merged_detections)
        
        return {
            'full_text': full_text,
            'detections': merged_detections,
            'layout': layout
        }
    
    def _detect_layout_regions(self, image: np.ndarray, detections: List[Dict]) -> Dict:
        """Detect document layout regions (header, body, footer)"""
        h, w = image.shape[:2]
        
        # Simple heuristic: top 20% = header, bottom 15% = footer, middle = body
        header_threshold = h * 0.20
        footer_threshold = h * 0.85
        
        header_texts = []
        body_texts = []
        footer_texts = []
        
        for det in detections:
            y_center = sum(p[1] for p in det['bbox']) / len(det['bbox'])
            
            if y_center < header_threshold:
                header_texts.append(det)
            elif y_center > footer_threshold:
                footer_texts.append(det)
            else:
                body_texts.append(det)
        
        return {
            'header': header_texts,
            'body': body_texts,
            'footer': footer_texts
        }


def load_image(image_path: str) -> np.ndarray:
    """Load and validate image"""
    if not image_path.lower().endswith('.png'):
        raise ValueError("Only PNG images are supported")
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    return image
