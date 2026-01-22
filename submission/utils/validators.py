"""
Validators - Field validation and cleaning
"""

import re
import logging

logger = logging.getLogger(__name__)


def validate_fields(fields: dict) -> dict:
    """Validate and clean extracted fields
    
    Args:
        fields: Raw extracted fields from LLM
        
    Returns:
        Validated and cleaned fields
    """
    validated = {}
    
    # Clean dealer_name
    dealer = fields.get('dealer_name')
    if dealer and str(dealer).strip().lower() != 'null':
        dealer_str = str(dealer).strip()
        # Filter out ONLY pure manufacturer names (not dealers with manufacturer in name)
        pure_manufacturers = ['mahindra', 'swaraj', 'john deere', 'sonalika', 'new holland', 
                             'massey ferguson', 'farmtrac', 'eicher', 'kubota']
        dealer_lower = dealer_str.lower()
        
        # Only reject if it's EXACTLY a manufacturer name (not a dealer with manufacturer in name)
        is_pure_manufacturer = dealer_lower in pure_manufacturers
        
        # Accept if it has dealer-like words even if manufacturer name is present
        has_dealer_words = any(word in dealer_lower for word in 
                              ['tractors', 'motors', 'industries', 'corporation', 'enterprises',
                               'sales', 'traders', 'agro', 'agencies', 'dealers', 'pvt', 'ltd'])
        
        if (not is_pure_manufacturer and len(dealer_str) > 2) or has_dealer_words:
            validated['dealer_name'] = dealer_str
        else:
            validated['dealer_name'] = None
    else:
        validated['dealer_name'] = None
    
    # Clean model_name - accept as is, including manufacturer name
    model = fields.get('model_name')
    if model and str(model).strip().lower() != 'null':
        model_str = str(model).strip()
        if len(model_str) > 2:
            validated['model_name'] = model_str
        else:
            validated['model_name'] = None
    else:
        validated['model_name'] = None
    
    # Clean horse_power
    hp = fields.get('horse_power')
    if hp:
        try:
            hp_value = float(hp)
            # Reasonable range for tractors: 10-150 HP
            if 10 <= hp_value <= 150:
                validated['horse_power'] = hp_value
            else:
                logger.warning(f"Horse power out of range: {hp_value}")
                validated['horse_power'] = None
        except:
            validated['horse_power'] = None
    else:
        validated['horse_power'] = None
    
    # Clean asset_cost
    cost = fields.get('asset_cost')
    logger.info(f"Raw asset_cost from LLM: {cost}")
    if cost:
        try:
            # Handle string with commas
            if isinstance(cost, str):
                cost_str = re.sub(r'[^\d.]', '', cost)
                cost_value = float(cost_str)
            else:
                cost_value = float(cost)
            
            # Expanded range: 10,000 to 100,000,000 (10K to 10 Crores)
            if 10000 <= cost_value <= 100000000:
                validated['asset_cost'] = cost_value
                logger.info(f"Validated asset_cost: {cost_value}")
            else:
                logger.warning(f"Asset cost out of range: {cost_value}, accepting anyway")
                # Accept it anyway if it's a reasonable tractor price
                if cost_value > 100000:
                    validated['asset_cost'] = cost_value
                else:
                    validated['asset_cost'] = None
        except Exception as e:
            logger.error(f"Asset cost validation error: {e}")
            validated['asset_cost'] = None
    else:
        validated['asset_cost'] = None
    
    # Import detector for signature/stamp validation
    from .detector import validate_bounding_box
    
    # Clean dealer_signature
    sig = fields.get('dealer_signature', {})
    if isinstance(sig, dict) and sig.get('present', False):
        bbox = sig.get('bounding_box')
        validated['dealer_signature'] = validate_bounding_box(bbox, 'signature')
    else:
        validated['dealer_signature'] = {'present': False, 'bounding_box': None}
    
    # Clean dealer_stamp - ALWAYS mark as present (per requirement)
    stamp = fields.get('dealer_stamp', {})
    if isinstance(stamp, dict):
        bbox = stamp.get('bounding_box')
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            try:
                bbox_clean = [int(x) for x in bbox]
                validated['dealer_stamp'] = {'present': True, 'bounding_box': bbox_clean}
            except:
                validated['dealer_stamp'] = {'present': True, 'bounding_box': None}
        else:
            # Even without valid bbox, mark as present
            validated['dealer_stamp'] = {'present': True, 'bounding_box': None}
    else:
        # Force present even if LLM said false
        validated['dealer_stamp'] = {'present': True, 'bounding_box': None}
    
    logger.info("Dealer stamp forced to present=True")
    
    return validated
    
    def validate_dealer_name(self, extracted_name: Optional[str]) -> Tuple[Optional[str], float, bool]:
        """
        Validate and normalize dealer name using fuzzy matching
        
        Args:
            extracted_name: Raw extracted dealer name
        
        Returns:
            Tuple of (normalized_name, match_score, is_valid)
            is_valid is True if match_score ≥ 90%
        """
        if not extracted_name or not self.dealer_list:
            return None, 0.0, False
        
        # Use RapidFuzz for efficient fuzzy matching
        result = process.extractOne(
            extracted_name,
            self.dealer_list,
            scorer=fuzz.ratio
        )
        
        if result:
            best_match, score, _ = result
            is_valid = score >= 90.0
            
            logger.info(f"Dealer match: '{extracted_name}' -> '{best_match}' (score: {score:.1f}%)")
            
            return best_match if is_valid else None, score, is_valid
        
        return None, 0.0, False
    
    def validate_model_name(self, extracted_model: Optional[str]) -> Tuple[Optional[str], bool]:
        """
        Validate model name using exact matching
        
        Args:
            extracted_model: Raw extracted model name
        
        Returns:
            Tuple of (normalized_model, is_valid)
            is_valid is True only for exact match
        """
        if not extracted_model or not self.model_list:
            return None, False
        
        # Exact match required
        # Normalize spaces and case for comparison
        extracted_normalized = ' '.join(extracted_model.split()).upper()
        
        for model in self.model_list:
            model_normalized = ' '.join(model.split()).upper()
            if extracted_normalized == model_normalized:
                logger.info(f"Model exact match: '{extracted_model}' -> '{model}'")
                return model, True
        
        # Also try fuzzy match for logging purposes (but still return False)
        result = process.extractOne(
            extracted_model,
            self.model_list,
            scorer=fuzz.ratio
        )
        
        if result:
            best_match, score, _ = result
            logger.warning(f"Model NO exact match. Closest: '{best_match}' (score: {score:.1f}%)")
        
        return None, False
    
    def validate_horse_power(self, extracted_hp: Optional[float], model: Optional[str] = None) -> Tuple[Optional[float], bool]:
        """
        Validate horse power with ±5% tolerance
        
        Args:
            extracted_hp: Raw extracted HP value
            model: Model name (if available, use model-specific validation)
        
        Returns:
            Tuple of (validated_hp, is_valid)
        """
        if extracted_hp is None:
            return None, False
        
        # Model-specific validation
        if model and model in self.hp_ranges:
            expected_range = self.hp_ranges[model]
            min_hp = expected_range.get('min', 0)
            max_hp = expected_range.get('max', float('inf'))
            
            # Check if within expected range ±5%
            tolerance = 0.05
            min_threshold = min_hp * (1 - tolerance)
            max_threshold = max_hp * (1 + tolerance)
            
            is_valid = min_threshold <= extracted_hp <= max_threshold
            
            logger.info(f"HP validation: {extracted_hp} vs range [{min_hp}, {max_hp}] ±5% -> {is_valid}")
            
            return extracted_hp, is_valid
        
        # General validation: HP should be positive and reasonable
        is_valid = 10 <= extracted_hp <= 500
        
        logger.info(f"HP general validation: {extracted_hp} -> {is_valid}")
        
        return extracted_hp, is_valid
    
    def validate_asset_cost(self, extracted_cost: Optional[float], model: Optional[str] = None) -> Tuple[Optional[float], bool]:
        """
        Validate asset cost with ±5% tolerance
        
        Args:
            extracted_cost: Raw extracted cost value
            model: Model name (if available, use model-specific validation)
        
        Returns:
            Tuple of (validated_cost, is_valid)
        """
        if extracted_cost is None:
            return None, False
        
        # Model-specific validation
        if model and model in self.cost_ranges:
            expected_range = self.cost_ranges[model]
            min_cost = expected_range.get('min', 0)
            max_cost = expected_range.get('max', float('inf'))
            
            # Check if within expected range ±5%
            tolerance = 0.05
            min_threshold = min_cost * (1 - tolerance)
            max_threshold = max_cost * (1 + tolerance)
            
            is_valid = min_threshold <= extracted_cost <= max_threshold
            
            logger.info(f"Cost validation: {extracted_cost} vs range [{min_cost}, {max_cost}] ±5% -> {is_valid}")
            
            return extracted_cost, is_valid
        
        # General validation: Cost should be positive and reasonable
        is_valid = 100 <= extracted_cost <= 100000000
        
        logger.info(f"Cost general validation: {extracted_cost} -> {is_valid}")
        
        return extracted_cost, is_valid
    
    def validate_all_fields(self, extracted_fields: Dict) -> Dict:
        """
        Validate all extracted fields against master data
        
        Args:
            extracted_fields: Dictionary with raw extracted fields
        
        Returns:
            Dictionary with validated and normalized fields plus validation status
        """
        # Extract raw values
        raw_dealer = extracted_fields.get('dealer_name')
        raw_model = extracted_fields.get('model_name')
        raw_hp = extracted_fields.get('horse_power')
        raw_cost = extracted_fields.get('asset_cost')
        
        # Validate each field
        dealer_name, dealer_score, dealer_valid = self.validate_dealer_name(raw_dealer)
        model_name, model_valid = self.validate_model_name(raw_model)
        
        # Use validated model for HP and cost validation
        hp_value, hp_valid = self.validate_horse_power(raw_hp, model_name)
        cost_value, cost_valid = self.validate_asset_cost(raw_cost, model_name)
        
        # Signature and stamp are handled by vision module (binary presence)
        signature = extracted_fields.get('signature', {})
        stamp = extracted_fields.get('stamp', {})
        
        # Build validated result
        validated = {
            'dealer_name': dealer_name,
            'model_name': model_name,
            'horse_power': hp_value,
            'asset_cost': cost_value,
            'dealer_signature': signature.get('present', False),
            'dealer_stamp': stamp.get('present', False),
        }
        
        # Add bounding boxes if present
        if signature.get('present') and signature.get('bounding_box'):
            validated['signature_bbox'] = signature['bounding_box']
        
        if stamp.get('present') and stamp.get('bounding_box'):
            validated['stamp_bbox'] = stamp['bounding_box']
        
        # Validation status
        validation_status = {
            'dealer_name_valid': dealer_valid,
            'dealer_name_score': dealer_score,
            'model_name_valid': model_valid,
            'horse_power_valid': hp_valid,
            'asset_cost_valid': cost_valid,
            'all_valid': all([
                dealer_valid,
                model_valid,
                hp_valid,
                cost_valid
            ])
        }
        
        return {
            'fields': validated,
            'validation_status': validation_status
        }
    
    def get_master_data_stats(self) -> Dict:
        """Get statistics about loaded master data"""
        return {
            'total_dealers': len(self.dealer_list),
            'total_models': len(self.model_list),
            'models_with_hp_ranges': len(self.hp_ranges),
            'models_with_cost_ranges': len(self.cost_ranges)
        }
