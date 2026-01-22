"""
Vision Detection Module - Signature and Stamp Detection
Uses YOLOv8 pretrained models for object detection with fine-tuning capability
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignatureStampDetector:
    """
    Detects dealer signatures and stamps using YOLOv8
    Returns binary presence + bounding boxes (IoU ≥ 0.5 required)
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to custom trained model, or None for pretrained
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        if model_path:
            try:
                logger.info(f"Loading custom model from {model_path}")
                self.model = YOLO(model_path)
            except Exception as e:
                logger.warning(f"Failed to load custom model: {e}")
                logger.info("Falling back to pretrained YOLOv8")
                self.model = YOLO('yolov8n.pt')  # Use nano for speed
        else:
            logger.info("Using pretrained YOLOv8n")
            self.model = YOLO('yolov8n.pt')
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect signatures and stamps in document
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Dictionary with signature and stamp detection results
        """
        # Preprocess for better detection
        preprocessed = self._preprocess_for_detection(image)
        
        # Run YOLO detection
        results = self.model(preprocessed, conf=self.confidence_threshold)
        
        # Extract signature and stamp detections
        signature_boxes = []
        stamp_boxes = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id] if hasattr(result, 'names') else str(cls_id)
                
                bbox = [x1, y1, x2, y2]
                detection = {
                    'bbox': bbox,
                    'confidence': conf,
                    'class': cls_name
                }
                
                # Classify based on visual characteristics
                # Since pretrained YOLO doesn't have signature/stamp classes,
                # we use heuristics + specialized detection
                roi = image[y1:y2, x1:x2]
                if self._is_signature(roi):
                    signature_boxes.append(detection)
                elif self._is_stamp(roi):
                    stamp_boxes.append(detection)
        
        # Use specialized detection if pretrained YOLO didn't find anything
        if not signature_boxes:
            signature_boxes = self._detect_signatures_heuristic(image)
        
        if not stamp_boxes:
            stamp_boxes = self._detect_stamps_heuristic(image)
        
        # Prepare output
        result = {
            'signature': {
                'present': len(signature_boxes) > 0,
                'bounding_box': signature_boxes[0]['bbox'] if signature_boxes else None,
                'confidence': signature_boxes[0]['confidence'] if signature_boxes else 0.0
            },
            'stamp': {
                'present': len(stamp_boxes) > 0,
                'bounding_box': stamp_boxes[0]['bbox'] if stamp_boxes else None,
                'confidence': stamp_boxes[0]['confidence'] if stamp_boxes else 0.0
            }
        }
        
        return result
    
    def _preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better detection"""
        # Resize if too large (YOLO works better with standard sizes)
        h, w = image.shape[:2]
        max_dim = 1280
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        return image
    
    def _is_signature(self, roi: np.ndarray) -> bool:
        """
        Heuristic to determine if ROI contains a signature
        Signatures: handwritten, irregular, flowing lines
        """
        if roi.size == 0:
            return False
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Signatures have:
        # 1. Low to medium pixel density
        # 2. Irregular contours (high variation in contour complexity)
        # 3. Horizontal aspect ratio preference
        
        h, w = gray.shape
        if w < 20 or h < 10:  # Too small
            return False
        
        aspect_ratio = w / h
        if not (1.5 < aspect_ratio < 5.0):  # Signatures are typically wide
            return False
        
        # Check pixel density
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        density = np.sum(binary == 255) / (w * h)
        
        if 0.05 < density < 0.35:  # Signature density range
            return True
        
        return False
    
    def _is_stamp(self, roi: np.ndarray) -> bool:
        """
        Heuristic to determine if ROI contains a stamp
        Stamps: circular/rectangular, uniform, high density
        """
        if roi.size == 0:
            return False
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        h, w = gray.shape
        if w < 30 or h < 30:  # Stamps are usually larger
            return False
        
        # Stamps tend to be more square
        aspect_ratio = w / h
        if not (0.7 < aspect_ratio < 1.5):
            return False
        
        # Check for circular patterns (common in stamps)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=10, maxRadius=min(h, w) // 2
        )
        
        if circles is not None:
            return True
        
        # Check color variance (stamps often have red/blue ink)
        if len(roi.shape) == 3:
            # Check for dominant red or blue channel
            b, g, r = cv2.split(roi)
            if np.mean(r) > np.mean(g) + 20 or np.mean(b) > np.mean(g) + 20:
                return True
        
        return False
    
    def _detect_signatures_heuristic(self, image: np.ndarray) -> List[Dict]:
        """
        Fallback signature detection using image processing
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for finding handwritten regions
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological operations to connect signature strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        signatures = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            
            if self._is_signature(roi):
                signatures.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': 0.6,  # Lower confidence for heuristic
                    'class': 'signature_heuristic'
                })
        
        return signatures
    
    def _detect_stamps_heuristic(self, image: np.ndarray) -> List[Dict]:
        """
        Fallback stamp detection using color and shape analysis
        """
        # Look for red/blue circular or rectangular regions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Red stamp detection (two ranges in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Blue stamp detection
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combine masks
        stamp_mask = cv2.bitwise_or(red_mask, blue_mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stamp_mask = cv2.morphologyEx(stamp_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(stamp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stamps = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Filter small noise
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            
            if self._is_stamp(roi):
                stamps.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': 0.65,  # Moderate confidence for heuristic
                    'class': 'stamp_heuristic'
                })
        
        return stamps
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union for two bounding boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2]
        
        Returns:
            IoU score (0.0 to 1.0)
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
