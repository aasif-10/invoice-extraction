"""
Confidence scoring and extraction notes generation
"""


def calculate_confidence(fields: dict) -> int:
    """Calculate confidence score based on fields extracted (capped at 95-99)
    
    Args:
        fields: Dictionary of extracted fields
        
    Returns:
        Confidence score (0-97)
    """
    score = 0
    total_fields = 6
    
    # Check each field
    if fields.get('dealer_name'):
        score += 1
    if fields.get('model_name'):
        score += 1
    if fields.get('horse_power'):
        score += 1
    if fields.get('asset_cost'):
        score += 1
    if fields.get('dealer_signature', {}).get('present'):
        score += 1
    if fields.get('dealer_stamp', {}).get('present'):
        score += 1
    
    # Calculate base confidence
    base_confidence = int((score / total_fields) * 100)
    
    # Cap at 95-99 to indicate inherent uncertainty in AI extraction
    if base_confidence == 100:
        return 97  # High confidence but not claiming perfection
    elif base_confidence >= 83:  # 5/6 fields
        return 95
    else:
        return base_confidence


def generate_extraction_notes(fields: dict) -> str:
    """Generate notes about missing or problematic fields
    
    Args:
        fields: Dictionary of extracted fields
        
    Returns:
        Extraction notes string
    """
    missing = []
    
    if not fields.get('dealer_name'):
        missing.append('dealer_name')
    if not fields.get('model_name'):
        missing.append('model_name')
    if not fields.get('horse_power'):
        missing.append('horse_power')
    if not fields.get('asset_cost'):
        missing.append('asset_cost')
    if not fields.get('dealer_signature', {}).get('present'):
        missing.append('dealer_signature')
    if not fields.get('dealer_stamp', {}).get('present'):
        missing.append('dealer_stamp')
    
    if not missing:
        return "All fields extracted successfully"
    elif len(missing) == 6:
        return "Complete extraction failure - no fields found"
    else:
        return f"Missing fields: {', '.join(missing)}"
