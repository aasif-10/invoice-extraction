"""
Consensus Module - Self-validation and confidence scoring without ground truth
Implements multi-strategy consensus to improve accuracy
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsensusValidator:
    """
    No-Ground-Truth validation strategy using:
    1. Multi-pass OCR consensus (already in OCR module)
    2. Cross-field logical consistency checks
    3. Document-level confidence aggregation
    4. Weak supervision via business rules
    """
    
    def __init__(self):
        """Initialize consensus validator"""
        self.business_rules = self._define_business_rules()
    
    def _define_business_rules(self) -> Dict:
        """
        Define weak supervision rules based on domain knowledge
        These rules help validate extractions without ground truth
        """
        return {
            # HP to cost correlation (higher HP -> higher cost typically)
            'hp_cost_correlation': {
                'enabled': True,
                'description': 'Higher HP should correlate with higher cost'
            },
            
            # Dealer name should appear in header
            'dealer_in_header': {
                'enabled': True,
                'description': 'Dealer name typically appears in document header'
            },
            
            # Model name should be alphanumeric
            'model_alphanumeric': {
                'enabled': True,
                'description': 'Model names are typically alphanumeric codes'
            },
            
            # Cost should be larger than HP numerically (sanity check)
            'cost_magnitude': {
                'enabled': True,
                'description': 'Cost should be significantly larger than HP value'
            },
            
            # Signature and stamp often appear together
            'signature_stamp_correlation': {
                'enabled': True,
                'description': 'Documents with signatures often have stamps too'
            }
        }
    
    def validate_cross_field_consistency(self, fields: Dict, validation_status: Dict) -> Dict:
        """
        Check logical consistency across extracted fields
        
        Args:
            fields: Validated field values
            validation_status: Field-level validation status
        
        Returns:
            Updated validation with consistency scores
        """
        consistency_checks = {}
        
        # Rule 1: HP-Cost correlation
        hp = fields.get('horse_power')
        cost = fields.get('asset_cost')
        
        if hp and cost:
            # Expected: higher HP models cost more
            # Basic heuristic: cost should be at least 1000x HP for tractors
            expected_min_cost = hp * 1000
            expected_max_cost = hp * 10000
            
            hp_cost_consistent = expected_min_cost <= cost <= expected_max_cost
            consistency_checks['hp_cost_correlation'] = {
                'consistent': hp_cost_consistent,
                'confidence_boost': 0.1 if hp_cost_consistent else -0.1
            }
        
        # Rule 2: Cost magnitude check
        if hp and cost:
            # Cost should be much larger than HP numerically
            magnitude_consistent = cost > hp * 50
            consistency_checks['cost_magnitude'] = {
                'consistent': magnitude_consistent,
                'confidence_boost': 0.05 if magnitude_consistent else -0.05
            }
        
        # Rule 3: Model name format
        model = fields.get('model_name')
        if model:
            # Model should be alphanumeric and reasonable length
            import re
            is_alphanumeric = bool(re.search(r'[A-Z0-9]', model))
            reasonable_length = 3 <= len(model) <= 20
            
            model_format_ok = is_alphanumeric and reasonable_length
            consistency_checks['model_format'] = {
                'consistent': model_format_ok,
                'confidence_boost': 0.05 if model_format_ok else -0.05
            }
        
        # Rule 4: Signature-Stamp correlation
        signature = fields.get('dealer_signature', False)
        stamp = fields.get('dealer_stamp', False)
        
        # If one is present, the other is more likely
        if signature or stamp:
            both_present = signature and stamp
            consistency_checks['signature_stamp'] = {
                'consistent': True,  # Any presence is good
                'confidence_boost': 0.1 if both_present else 0.05
            }
        
        return consistency_checks
    
    def compute_document_confidence(self, 
                                    validation_status: Dict, 
                                    consistency_checks: Dict,
                                    ocr_confidence: float,
                                    vision_confidence: float) -> float:
        """
        Compute overall document-level confidence score
        
        Args:
            validation_status: Field validation results
            consistency_checks: Cross-field consistency results
            ocr_confidence: Average OCR confidence from multi-pass
            vision_confidence: Average vision detection confidence
        
        Returns:
            Document-level confidence (0.0 to 1.0)
        """
        # Base confidence from field validation
        field_confidences = []
        
        # Dealer name (use match score)
        if validation_status.get('dealer_name_valid'):
            dealer_score = validation_status.get('dealer_name_score', 0) / 100.0
            field_confidences.append(dealer_score)
        else:
            field_confidences.append(0.0)
        
        # Model name (binary: exact match or not)
        field_confidences.append(1.0 if validation_status.get('model_name_valid') else 0.0)
        
        # HP (binary validation)
        field_confidences.append(1.0 if validation_status.get('horse_power_valid') else 0.5)
        
        # Cost (binary validation)
        field_confidences.append(1.0 if validation_status.get('asset_cost_valid') else 0.5)
        
        # Signature and stamp (from vision confidence)
        field_confidences.append(vision_confidence)
        field_confidences.append(vision_confidence)
        
        # Average field confidence
        avg_field_confidence = np.mean(field_confidences) if field_confidences else 0.0
        
        # Apply consistency boosts
        consistency_boost = sum(
            check.get('confidence_boost', 0) 
            for check in consistency_checks.values()
        )
        
        # Combine: 60% field validation + 20% OCR + 10% vision + 10% consistency
        document_confidence = (
            0.6 * avg_field_confidence +
            0.2 * ocr_confidence +
            0.1 * vision_confidence +
            0.1 * (0.5 + consistency_boost)  # Baseline 0.5 + boosts
        )
        
        # Clamp to [0, 1]
        document_confidence = max(0.0, min(1.0, document_confidence))
        
        logger.info(f"Document confidence: {document_confidence:.3f} "
                   f"(fields={avg_field_confidence:.2f}, OCR={ocr_confidence:.2f}, "
                   f"vision={vision_confidence:.2f}, consistency={consistency_boost:+.2f})")
        
        return document_confidence
    
    def pseudo_label_bootstrap(self, fields: Dict, confidence: float) -> Dict:
        """
        Generate pseudo-labels for weak supervision
        High-confidence extractions can be used as training data (if needed)
        
        Args:
            fields: Extracted fields
            confidence: Document confidence
        
        Returns:
            Pseudo-labeled data (for potential model improvement)
        """
        # Only generate pseudo-labels for high-confidence documents
        if confidence < 0.85:
            return None
        
        # Create pseudo-label entry
        pseudo_label = {
            'fields': fields.copy(),
            'confidence': confidence,
            'source': 'consensus_validation',
            'usable_for_training': True
        }
        
        logger.info(f"Generated pseudo-label with confidence {confidence:.3f}")
        
        return pseudo_label
    
    def detect_anomalies(self, fields: Dict) -> List[str]:
        """
        Detect potential anomalies in extracted data
        
        Args:
            fields: Extracted fields
        
        Returns:
            List of anomaly descriptions
        """
        anomalies = []
        
        # Check for missing critical fields
        critical_fields = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        for field in critical_fields:
            if not fields.get(field):
                anomalies.append(f"Missing critical field: {field}")
        
        # Check for unrealistic values
        hp = fields.get('horse_power')
        if hp and (hp < 5 or hp > 1000):
            anomalies.append(f"Unusual horse power: {hp} (expected 5-1000)")
        
        cost = fields.get('asset_cost')
        if cost and (cost < 10 or cost > 50000000):
            anomalies.append(f"Unusual asset cost: {cost} (expected 10-50M)")
        
        # Check for signature/stamp presence
        if not fields.get('dealer_signature') and not fields.get('dealer_stamp'):
            anomalies.append("Neither signature nor stamp detected")
        
        return anomalies
    
    def explain_confidence(self, 
                          validation_status: Dict, 
                          consistency_checks: Dict,
                          confidence: float) -> str:
        """
        Generate human-readable explanation of confidence score
        
        Args:
            validation_status: Validation results
            consistency_checks: Consistency check results
            confidence: Final confidence score
        
        Returns:
            Explanation string
        """
        explanations = []
        
        # Field validation
        valid_fields = sum([
            validation_status.get('dealer_name_valid', False),
            validation_status.get('model_name_valid', False),
            validation_status.get('horse_power_valid', False),
            validation_status.get('asset_cost_valid', False)
        ])
        
        explanations.append(f"Field validation: {valid_fields}/4 fields valid")
        
        # Dealer match score
        dealer_score = validation_status.get('dealer_name_score', 0)
        explanations.append(f"Dealer match score: {dealer_score:.1f}%")
        
        # Consistency
        consistent_checks = sum(
            1 for check in consistency_checks.values() 
            if check.get('consistent', False)
        )
        total_checks = len(consistency_checks)
        
        if total_checks > 0:
            explanations.append(f"Consistency checks: {consistent_checks}/{total_checks} passed")
        
        # Overall
        confidence_label = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.6 else "LOW"
        explanations.append(f"Overall confidence: {confidence:.3f} ({confidence_label})")
        
        return " | ".join(explanations)
    
    def run_full_consensus(self, 
                          fields: Dict, 
                          validation_status: Dict,
                          ocr_confidence: float,
                          vision_confidence: float) -> Dict:
        """
        Run complete consensus validation pipeline
        
        Args:
            fields: Extracted and validated fields
            validation_status: Field-level validation status
            ocr_confidence: OCR confidence
            vision_confidence: Vision confidence
        
        Returns:
            Complete consensus result with confidence and explanations
        """
        # Cross-field consistency
        consistency_checks = self.validate_cross_field_consistency(fields, validation_status)
        
        # Document-level confidence
        document_confidence = self.compute_document_confidence(
            validation_status, consistency_checks, ocr_confidence, vision_confidence
        )
        
        # Anomaly detection
        anomalies = self.detect_anomalies(fields)
        
        # Confidence explanation
        explanation = self.explain_confidence(
            validation_status, consistency_checks, document_confidence
        )
        
        # Pseudo-labeling (if high confidence)
        pseudo_label = self.pseudo_label_bootstrap(fields, document_confidence)
        
        return {
            'confidence': document_confidence,
            'consistency_checks': consistency_checks,
            'anomalies': anomalies,
            'explanation': explanation,
            'pseudo_label': pseudo_label
        }
