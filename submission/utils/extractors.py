"""
Field Extractors - Semantic extraction of required fields from document text
Uses NER, pattern matching, and layout reasoning for generalized extraction
"""

import re
import spacy
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FieldExtractor:
    """
    Extracts the six required fields using semantic understanding:
    - Dealer Name (text + fuzzy match)
    - Model Name (text + exact match)
    - Horse Power (numeric + semantic context)
    - Asset Cost (numeric + semantic context)
    - Dealer Signature (handled by vision module)
    - Dealer Stamp (handled by vision module)
    """
    
    def __init__(self):
        """Initialize NER and pattern matchers"""
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("Loaded spaCy NER model")
        except:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for field extraction"""
        # Horse Power patterns (HP, BHP, bhp, horsepower, etc.)
        self.hp_patterns = [
            re.compile(r'(\d+\.?\d*)\s*(?:hp|HP|bhp|BHP|h\.p\.|horsepower)', re.IGNORECASE),
            re.compile(r'(?:hp|HP|bhp|BHP|horsepower)[:\s]*(\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'(?:power|engine)[:\s]*(\d+\.?\d*)\s*(?:hp|HP|bhp|BHP)', re.IGNORECASE),
        ]
        
        # Asset Cost patterns (total, amount, cost, etc.)
        self.cost_patterns = [
            re.compile(r'(?:total|amount|cost|price)[:\s]*[₹$£€]?\s*(\d+[,\d]*\.?\d*)', re.IGNORECASE),
            re.compile(r'[₹$£€]\s*(\d+[,\d]*\.?\d*)\s*(?:total|amount|cost)?', re.IGNORECASE),
            re.compile(r'(?:grand\s*total|net\s*total)[:\s]*[₹$£€]?\s*(\d+[,\d]*\.?\d*)', re.IGNORECASE),
            re.compile(r'(?:asset\s*cost|vehicle\s*cost)[:\s]*[₹$£€]?\s*(\d+[,\d]*\.?\d*)', re.IGNORECASE),
        ]
        
        # Dealer name patterns
        self.dealer_patterns = [
            re.compile(r'(?:dealer|sold\s*by|authorized\s*dealer)[:\s]*([A-Z][A-Za-z\s&\.]+)', re.IGNORECASE),
            re.compile(r'(?:from|by)[:\s]*([A-Z][A-Za-z\s&\.]+)\s*(?:dealer|motors|automobiles)', re.IGNORECASE),
        ]
        
        # Model name patterns (vehicle model)
        self.model_patterns = [
            re.compile(r'(?:model|vehicle\s*model)[:\s]*([A-Z0-9][A-Za-z0-9\s\-]+)', re.IGNORECASE),
            re.compile(r'(?:tractor|vehicle|machine)[:\s]*([A-Z0-9][A-Za-z0-9\s\-]+)', re.IGNORECASE),
        ]
    
    def extract_all_fields(self, ocr_data: Dict, layout: Dict) -> Dict:
        """
        Extract all required fields from OCR data
        
        Args:
            ocr_data: Dictionary with 'full_text' and 'detections'
            layout: Layout regions (header, body, footer)
        
        Returns:
            Dictionary with extracted field candidates
        """
        full_text = ocr_data.get('full_text', '')
        detections = ocr_data.get('detections', [])
        
        # Extract each field
        dealer_name = self.extract_dealer_name(full_text, layout, detections)
        model_name = self.extract_model_name(full_text, layout, detections)
        horse_power = self.extract_horse_power(full_text, detections)
        asset_cost = self.extract_asset_cost(full_text, detections)
        
        return {
            'dealer_name': dealer_name,
            'model_name': model_name,
            'horse_power': horse_power,
            'asset_cost': asset_cost
        }
    
    def extract_dealer_name(self, text: str, layout: Dict, detections: List[Dict]) -> Optional[str]:
        """
        Extract dealer name using NER + pattern matching
        Priority: header region, then body
        """
        candidates = []
        
        # Strategy 1: Pattern matching
        for pattern in self.dealer_patterns:
            matches = pattern.findall(text)
            for match in matches:
                candidates.append({
                    'value': match.strip(),
                    'confidence': 0.8,
                    'method': 'pattern'
                })
        
        # Strategy 2: NER - find ORGANIZATION entities in header
        if self.nlp:
            header_text = ' '.join([d['text'] for d in layout.get('header', [])])
            doc = self.nlp(header_text)
            
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    candidates.append({
                        'value': ent.text,
                        'confidence': 0.75,
                        'method': 'ner'
                    })
        
        # Strategy 3: Look for capitalized names in header (common for dealer names)
        header_detections = layout.get('header', [])
        for det in header_detections:
            text = det['text']
            # Check if it's a capitalized name (2+ words, title case)
            if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text):
                candidates.append({
                    'value': text,
                    'confidence': 0.65,
                    'method': 'header_capitalized'
                })
        
        # Return highest confidence candidate
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            return best['value']
        
        return None
    
    def extract_model_name(self, text: str, layout: Dict, detections: List[Dict]) -> Optional[str]:
        """
        Extract model name using pattern matching
        Look for alphanumeric model codes (e.g., "MF 5710", "JD 5050D")
        """
        candidates = []
        
        # Strategy 1: Explicit pattern matching
        for pattern in self.model_patterns:
            matches = pattern.findall(text)
            for match in matches:
                candidates.append({
                    'value': match.strip(),
                    'confidence': 0.85,
                    'method': 'pattern'
                })
        
        # Strategy 2: Find alphanumeric codes (common model format)
        # Pattern: 2-4 letters followed by space and 3-5 digits, optionally followed by letter
        model_code_pattern = re.compile(r'\b([A-Z]{2,4}[\s\-]?\d{3,5}[A-Z]?)\b')
        matches = model_code_pattern.findall(text)
        for match in matches:
            candidates.append({
                'value': match,
                'confidence': 0.75,
                'method': 'code_pattern'
            })
        
        # Strategy 3: Look in body for product identifiers
        body_text = ' '.join([d['text'] for d in layout.get('body', [])])
        
        # Find words that look like model names (mixed alphanumeric)
        words = body_text.split()
        for i, word in enumerate(words):
            if re.match(r'^[A-Z0-9]+$', word) and len(word) >= 4 and len(word) <= 10:
                # Check context - if preceded by keywords
                context = ' '.join(words[max(0, i-2):i]).lower()
                if any(kw in context for kw in ['model', 'type', 'series', 'variant']):
                    candidates.append({
                        'value': word,
                        'confidence': 0.70,
                        'method': 'context'
                    })
        
        # Return highest confidence candidate
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            return best['value']
        
        return None
    
    def extract_horse_power(self, text: str, detections: List[Dict]) -> Optional[float]:
        """
        Extract horse power as numeric value
        Look for HP, BHP, horsepower mentions with associated numbers
        """
        candidates = []
        
        # Use compiled patterns
        for pattern in self.hp_patterns:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    value = float(match.replace(',', ''))
                    # Sanity check: HP typically between 10 and 500 for tractors/vehicles
                    if 10 <= value <= 500:
                        candidates.append({
                            'value': value,
                            'confidence': 0.9,
                            'method': 'pattern'
                        })
                except ValueError:
                    continue
        
        # Strategy 2: Semantic proximity - find numbers near "power" or "engine"
        if self.nlp:
            doc = self.nlp(text)
            for token in doc:
                if token.like_num and token.text.replace('.', '').replace(',', '').isdigit():
                    try:
                        num_value = float(token.text.replace(',', ''))
                        
                        # Check surrounding context
                        context_window = doc[max(0, token.i - 3):min(len(doc), token.i + 3)]
                        context_text = ' '.join([t.text.lower() for t in context_window])
                        
                        if any(kw in context_text for kw in ['hp', 'bhp', 'power', 'horsepower']):
                            if 10 <= num_value <= 500:
                                candidates.append({
                                    'value': num_value,
                                    'confidence': 0.75,
                                    'method': 'semantic'
                                })
                    except ValueError:
                        continue
        
        # Return highest confidence candidate
        if candidates:
            # Prefer higher confidence, then higher value (if multiple candidates)
            best = max(candidates, key=lambda x: (x['confidence'], x['value']))
            return best['value']
        
        return None
    
    def extract_asset_cost(self, text: str, detections: List[Dict]) -> Optional[float]:
        """
        Extract asset cost as numeric value
        Look for total, cost, amount mentions with currency
        """
        candidates = []
        
        # Use compiled patterns
        for pattern in self.cost_patterns:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    # Remove commas and parse
                    value = float(match.replace(',', ''))
                    # Sanity check: cost typically > 100 and < 100M
                    if 100 <= value <= 100000000:
                        candidates.append({
                            'value': value,
                            'confidence': 0.9,
                            'method': 'pattern'
                        })
                except ValueError:
                    continue
        
        # Strategy 2: Find largest number in document (often the total)
        # But only if it has currency symbol nearby
        numbers_with_currency = re.findall(r'[₹$£€]\s*(\d+[,\d]*\.?\d*)', text)
        for match in numbers_with_currency:
            try:
                value = float(match.replace(',', ''))
                if 100 <= value <= 100000000:
                    candidates.append({
                        'value': value,
                        'confidence': 0.70,
                        'method': 'currency'
                    })
            except ValueError:
                continue
        
        # Strategy 3: Look for "Total" or "Grand Total" labels and find nearby number
        if self.nlp:
            doc = self.nlp(text)
            for i, token in enumerate(doc):
                if token.text.lower() in ['total', 'amount', 'cost']:
                    # Look ahead for number
                    for j in range(i + 1, min(i + 5, len(doc))):
                        if doc[j].like_num:
                            try:
                                value = float(doc[j].text.replace(',', ''))
                                if 100 <= value <= 100000000:
                                    candidates.append({
                                        'value': value,
                                        'confidence': 0.85,
                                        'method': 'label_proximity'
                                    })
                            except ValueError:
                                continue
        
        # Return highest confidence candidate (prefer largest value for ties)
        if candidates:
            # For cost, we want highest confidence and typically the largest value
            best = max(candidates, key=lambda x: (x['confidence'], x['value']))
            return best['value']
        
        return None
    
    def extract_with_spatial_context(self, detections: List[Dict], field_type: str) -> Optional[any]:
        """
        Extract field using spatial/layout reasoning
        Useful when text patterns fail
        
        Args:
            detections: List of OCR detections with bbox info
            field_type: Type of field to extract
        
        Returns:
            Extracted value or None
        """
        # Group detections into table-like structures
        # Find values based on column/row alignment
        # This is a more advanced extraction for structured documents
        
        # TODO: Implement table structure detection if needed
        # For now, return None (fallback to pattern matching)
        return None
