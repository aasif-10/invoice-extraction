"""
Signature and stamp detection utilities
"""

import logging

logger = logging.getLogger(__name__)


def validate_bounding_box(bbox: list, field_name: str) -> dict:
    """Validate and clean bounding box (lenient - prefer accepting over rejecting)
    
    Args:
        bbox: Bounding box list [x, y, width, height]
        field_name: Name of field for logging
        
    Returns:
        Dictionary with present flag and validated bounding box
    """
    if bbox and isinstance(bbox, list) and len(bbox) == 4:
        try:
            bbox_clean = [int(x) for x in bbox]
            # Be lenient - accept any bounding box with at least some positive values
            # Only reject pure [0,0,0,0]
            if bbox_clean == [0, 0, 0, 0]:
                logger.warning(f"{field_name} has [0,0,0,0] bounding box - but marking as present anyway")
                # Even [0,0,0,0] is accepted - be very lenient
                return {
                    'present': True,
                    'bounding_box': bbox_clean
                }
            else:
                logger.info(f"Valid {field_name} bounding box: {bbox_clean}")
                return {
                    'present': True,
                    'bounding_box': bbox_clean
                }
        except Exception as e:
            logger.warning(f"Error parsing {field_name} bbox, but marking present anyway: {e}")
            # Even on error, mark as present if LLM said so
            return {'present': True, 'bounding_box': None}
    else:
        # No bbox but still mark present if we got here
        logger.info(f"No valid {field_name} bbox, marking as present with no coordinates")
        return {'present': True, 'bounding_box': None}
