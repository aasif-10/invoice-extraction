"""
Field extraction using Vision LLM
"""

import json
import requests
import logging

logger = logging.getLogger(__name__)


def extract_with_vision_llm(image_base64: str, llm_url: str, model: str = "llama3.2-vision") -> dict:
    """Use vision LLM to intelligently extract asset invoice fields from image
    
    Args:
        image_base64: Base64 encoded image
        llm_url: Ollama API endpoint
        model: Vision model name
        
    Returns:
        Dictionary with extracted fields
    """
    
    prompt = """Analyze this invoice/bill image and extract information. This could be ANY type of invoice format (formal, handwritten, mixed languages, poor quality).

IMPORTANT: You are analyzing a REAL invoice image. Extract the 6 required fields regardless of:
- Document format (typed, handwritten, mixed)
- Language (English, Hindi, regional, mixed)
- Quality (clear, faded, scanned, photographed)
- Layout (structured table, unstructured, varied formats)

Extract these 6 fields with MAXIMUM effort:

1. DEALER_NAME: The company/person selling the equipment. THIS IS CRITICAL:
   - Look EVERYWHERE in header/top section for business names
   - Accept: typed names, handwritten names, stamps with company name
   - Look for: "Sold by", "Dealer", "Seller", letterhead, any capitalized business name
   - Common patterns: "[Name] TRACTORS", "[Name] MOTORS", "[Name] AGRO", "[Name] ENTERPRISES"
   - Can be in English, Hindi, or regional language - extract as-is
   - If multiple names, pick the dealer/seller (not manufacturer)
   - NEVER return null - if unclear, return the most prominent business name you see
   - Examples: "SABAR AGROTECH", "RAM KUMAR MOTORS", "श्री कृष्णा ट्रैक्टर्स"

2. MODEL_NAME: The complete tractor/equipment model INCLUDING manufacturer. CRITICAL:
   - Look in: item description, specifications, model field, product details
   - INCLUDE manufacturer name: "SONALIKA DI 35 RX PS", "Mahindra 475 DI", "TAFE MF 241"
   - Accept handwritten model names, typed, in tables, anywhere
   - Can be in English or transliterated - extract EXACTLY as written
   - Common patterns: "[Brand] [Model Code]", "[Brand] [Series] [Number]"
   - If unclear, look for alphanumeric codes with tractor brand names
   - Examples: "TAFE MF 241", "Mahindra Arjun 605", "Sonalika DI 35 RX PS"

3. HORSE_POWER: Engine power rating. CRITICAL:
   - Look for: "HP", "H.P.", "BHP", "Horse Power", "अश्वशक्ति"
   - Find the NUMBER near these keywords (e.g., 42, 47, 50)
   - Can be in specifications, engine details, technical section
   - Accept both typed and handwritten numbers
   - DO NOT confuse with model numbers (DI 35 is NOT 35 HP)
   - Return only the numeric value: 42.0, 47.0, 50.0
   - If range given (40-45 HP), use middle value: 42.5

4. ASSET_COST: Total invoice amount. CRITICAL:
   - Look for: "Total", "Grand Total", "Net Amount", "Amount Payable", "कुल राशि"
   - Find the LARGEST number on the invoice (usually the total)
   - Can be typed, handwritten, in table footer, anywhere
   - Handle formats: 8,30,000 OR 830000 OR 8.3 Lakhs OR ₹830000
   - If in Lakhs/Lacs, convert: 8.3 Lakhs = 830000
   - Read ALL digits carefully - don't miss zeros
   - Return raw number: 830000, 1250000, 450000
   - Accept both clear and unclear handwritten amounts

5. DEALER_SIGNATURE: ANY handwritten signature/mark:
   - Look VERY CAREFULLY at bottom section, right side, near "Authorized Signatory"
   - Accept: clear signatures, messy scribbles, cursive, print-style, ANY handwriting
   - Different styles: English cursive, Hindi script, simple marks, initials
   - Can be faint, unclear, or bold - still count it
   - ALWAYS mark present=true if you see ANY hand-drawn marks
   - Estimate position: [x, y, width, height] in pixels from top-left (0,0)
   - Use realistic coordinates based on actual image size

6. DEALER_STAMP: ANY stamp/seal mark:
   - Look for: circular stamps, rectangular seals, company impressions
   - Any color: red, blue, black, purple, faint
   - Can be: official stamps, rubber stamps, embossed seals, ink impressions
   - Check: near signature, bottom section, anywhere on document
   - ALWAYS mark present=true if you see ANY stamp-like mark
   - Estimate position: [x, y, width, height] in pixels
   - Even faint or partial stamps count

Return ONLY this JSON structure (no markdown, no code blocks):

{
  "dealer_name": "string or null",
  "model_name": "string or null",
  "horse_power": number or null,
  "asset_cost": number or null,
  "dealer_signature": {
    "present": true or false,
    "bounding_box": [x, y, width, height] or null
  },
  "dealer_stamp": {
    "present": true or false,
    "bounding_box": [x, y, width, height] or null
  }
}

CRITICAL RULES:
- Work with ANY invoice format - structured, unstructured, handwritten, typed, mixed
- Handle multiple languages - English, Hindi, regional, transliterated
- dealer_name is MANDATORY - never null unless truly no text readable
- model_name should include manufacturer (full as written)
- For numbers: read carefully, handle Indian formats (lakhs), don't miss digits
- For signature/stamp: be EXTREMELY generous - if ANY mark exists, mark present=true
- Bounding boxes: provide REALISTIC coordinates based on actual position in image (e.g., [x, y, width, height] where signature/stamp appears)
  - NEVER use [0, 0, 0, 0] - this is invalid
  - Look at actual image dimensions and estimate where signature/stamp is located
  - Typical invoice image size: 1200x1600 pixels
  - Signature usually bottom-right: estimate [850, 1400, 250, 120]
  - Stamp usually bottom-left or near signature: estimate [150, 1450, 200, 150]
  - If you cannot determine exact location but mark is present: use these typical estimates
  - Better to estimate than return [0,0,0,0] or null
- Return null ONLY if field absolutely does not exist (except signature/stamp - be generous)
- Return pure JSON only - no explanations, no markdown"""
    
    try:
        response = requests.post(
            llm_url,
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result.get('response', '').strip()
            
            logger.info(f"Vision LLM Response: {llm_response[:300]}...")
            
            # Extract JSON from response
            if '{' in llm_response and '}' in llm_response:
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                json_str = llm_response[json_start:json_end]
                
                # Try parsing with error recovery
                try:
                    fields = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error: {e}. Attempting cleanup...")
                    # Clean up common issues
                    import re
                    json_str = re.sub(r"[''`]", '', json_str)
                    try:
                        fields = json.loads(json_str)
                    except:
                        logger.error("JSON parsing failed after cleanup")
                        return _get_empty_fields()
                
                logger.info("Vision LLM extraction successful")
                return fields
            else:
                logger.error("No JSON found in vision LLM response")
                return _get_empty_fields()
        else:
            logger.error(f"Vision LLM request failed: {response.status_code}")
            return _get_empty_fields()
            
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Is it running?")
        return _get_empty_fields()
    except Exception as e:
        logger.error(f"Vision LLM extraction failed: {e}")
        return _get_empty_fields()


def _get_empty_fields() -> dict:
    """Return empty field structure"""
    return {
        "dealer_name": None,
        "model_name": None,
        "horse_power": None,
        "asset_cost": None,
        "dealer_signature": {"present": False, "bounding_box": None},
        "dealer_stamp": {"present": False, "bounding_box": None}
    }
