import cv2
import numpy as np
import os
import re

def preprocess_image(image_path):
    """Preprocess the image for better OCR and detection"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Basic preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    return thresh, image

def extract_date(text):
    """Extract date from text using regex"""
    # Common date formats
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{1,2}-\d{1,2}-\d{4}',
        r'\d{1,2}\.\d{1,2}\.\d{4}',
        r'\d{4}/\d{1,2}/\d{1,2}',
        r'\d{4}-\d{1,2}-\d{1,2}',
        r'\d{2}/\d{2}/\d{2}'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None

def extract_invoice_number(text):
    """Extract invoice number using regex with support for P.O. numbers"""
    # Common invoice number patterns
    invoice_patterns = [
        r'invoice\s*#?\s*(\w+[-]?\w+)',
        r'invoice\s*no[.:]?\s*(\w+[-]?\w+)',
        r'invoice\s*number[.:]?\s*(\w+[-]?\w+)',
        r'inv\s*#?\s*(\w+[-]?\w+)',
        r'#\s*(\w+[-]?\w+)',
        r'p\.?o\.?\s*#?\s*([0-9]+)',
        r'p\.?o\.?\s*number\s*([0-9]+)',
        r'export references.*?(\d+\s+\d+\s+\d+)'
    ]
    
    for pattern in invoice_patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip()
    return None

def extract_line_items(image, reader, boxes=None, texts=None):
    """
    Extract line items from an invoice image by analyzing tabular structures
    
    Args:
        image: The invoice image
        reader: EasyOCR reader instance
        boxes: Optional pre-extracted text boxes
        texts: Optional pre-extracted text contents
    
    Returns:
        List of line items with their properties
    """
    # Run OCR if boxes and texts are not provided
    if boxes is None or texts is None:
        results = reader.readtext(image)
        texts = [result[1] for result in results]
        boxes = [result[0] for result in results]
    
    # Detect potential table structure
    h, w = image.shape[:2]
    
    # Group text boxes by Y-coordinate to detect rows
    rows = {}
    for i, (text, box) in enumerate(zip(texts, boxes)):
        # Get center Y coordinate of the box
        y_center = sum(corner[1] for corner in box) / 4
        # Group by approximate rows (10px tolerance)
        y_bin = int(y_center / 10) * 10
        
        if y_bin not in rows:
            rows[y_bin] = []
        
        # Get center X coordinate
        x_center = sum(corner[0] for corner in box) / 4
        
        # Add to row with position info
        rows[y_bin].append({
            'text': text,
            'box': box,
            'x': x_center,
            'y': y_center,
            'width': max(corner[0] for corner in box) - min(corner[0] for corner in box)
        })
    
    # Sort rows by Y position (top to bottom)
    sorted_rows = sorted(rows.items())
    
    # Find potential header row (contains keywords like qty, description, price, amount)
    header_row = None
    header_row_idx = None
    header_keywords = ['qty', 'quantity', 'description', 'item', 'unit', 'price', 'amount', 'total']
    
    for idx, (y, row) in enumerate(sorted_rows):
        row_text = " ".join([item['text'].lower() for item in row])
        if any(keyword in row_text for keyword in header_keywords):
            # Count matches to ensure it's really a header row
            matches = sum(1 for keyword in header_keywords if keyword in row_text)
            if matches >= 2:  # At least two header keywords found
                header_row = row
                header_row_idx = idx
                break
    
    # If no header row found, try to infer columns from the data
    if header_row is None:
        # Look for rows with numeric values that might be prices
        price_pattern = r'[\d,]+\.\d{2}'
        potential_item_rows = []
        
        for y, row in sorted_rows:
            # Check if row contains price-like values
            price_found = any(re.search(price_pattern, item['text']) for item in row)
            if price_found and len(row) >= 2:  # At least two columns
                potential_item_rows.append((y, row))
        
        if potential_item_rows:
            # Analyze column positions across all potential item rows
            all_x_positions = []
            for _, row in potential_item_rows:
                all_x_positions.extend([item['x'] for item in row])
            
            # Use clustering to identify column positions
            column_positions = []
            if all_x_positions:
                # Sort positions
                all_x_positions.sort()
                
                # Find clusters (gaps larger than 50px indicate new column)
                current_cluster = [all_x_positions[0]]
                for pos in all_x_positions[1:]:
                    if pos - current_cluster[-1] > 50:
                        # Calculate average position for this cluster
                        column_positions.append(sum(current_cluster) / len(current_cluster))
                        # Start new cluster
                        current_cluster = [pos]
                    else:
                        current_cluster.append(pos)
                
                # Add last cluster
                if current_cluster:
                    column_positions.append(sum(current_cluster) / len(current_cluster))
            
            # Assign column types based on content
            columns = []
            for col_idx, col_pos in enumerate(column_positions):
                # Check content across rows to determine column type
                col_values = []
                for _, row in potential_item_rows:
                    # Find closest item to this column position
                    closest = min(row, key=lambda item: abs(item['x'] - col_pos))
                    if abs(closest['x'] - col_pos) < 100:  # Within reasonable distance
                        col_values.append(closest['text'])
                
                # Determine column type based on content
                if col_values:
                    # Check if column contains numbers that look like quantities
                    if all(re.match(r'^\d+$', val) for val in col_values if val.strip()):
                        columns.append(('quantity', col_pos))
                    # Check if column contains currency/price values
                    elif any(re.search(price_pattern, val) for val in col_values):
                        if col_idx == len(column_positions) - 1:
                            columns.append(('amount', col_pos))
                        else:
                            columns.append(('price', col_pos))
                    # If it's the second column, likely description
                    elif col_idx == 1:
                        columns.append(('description', col_pos))
                    # Default to description for other columns
                    else:
                        columns.append(('description', col_pos))
            
            # Extract line items based on inferred columns
            line_items = []
            for _, row in potential_item_rows:
                item = {}
                for col_type, col_pos in columns:
                    # Find closest item to this column position
                    closest = min(row, key=lambda item: abs(item['x'] - col_pos))
                    if abs(closest['x'] - col_pos) < 100:  # Within reasonable distance
                        item[col_type] = closest['text']
                
                # Skip rows that don't look like line items or contain non-line item keywords
                non_line_item_keywords = ['subtotal', 'total', 'tax', 'gst', 'terms', 'conditions', 'notes', 'balance due', 'gss']
                row_text = ' '.join(item.values()).lower()
                if item and not any(keyword in row_text for keyword in non_line_item_keywords):
                    # Additional validation: At least one numeric value must be present in a line item
                    has_numeric = any(re.search(r'\d+', value) for value in item.values())
                    if has_numeric:
                        line_items.append(item)
            
            return line_items
    
    # If header row found, identify columns
    columns = []
    if header_row:
        for item in header_row:
            text_lower = item['text'].lower()
            if any(keyword in text_lower for keyword in ['qty', 'quantity']):
                columns.append(('quantity', item['x']))
            elif any(keyword in text_lower for keyword in ['desc', 'description', 'item']):
                columns.append(('description', item['x']))
            elif any(keyword in text_lower for keyword in ['price', 'unit']):
                columns.append(('price', item['x']))
            elif any(keyword in text_lower for keyword in ['amount', 'total']):
                columns.append(('amount', item['x']))
        
        # Extract line items
        line_items = []
        if header_row_idx is not None:
            # These keywords shouldn't appear in line items
            non_line_item_keywords = ['subtotal', 'total', 'tax', 'gst', 'terms', 'conditions', 'notes', 'balance due', 'gss']
            
            for y, row in sorted_rows[header_row_idx+1:]:
                # Skip rows that might be totals, empty, or contain terms/notes/etc.
                row_text = " ".join([item['text'].lower() for item in row])
                if any(word in row_text for word in non_line_item_keywords):
                    continue
                    
                if not row:  # Skip empty rows
                    continue
                
                # Create line item
                item = {}
                for col_type, col_x in columns:
                    # Find text closest to this column
                    closest = min(row, key=lambda r: abs(r['x'] - col_x))
                    if abs(closest['x'] - col_x) < 100:  # Within reasonable distance
                        item[col_type] = closest['text']
                
                # Only add if we have at least one value and it contains numeric content
                if item:
                    # Check for numeric content in the item
                    has_numeric = any(re.search(r'\d+', value) for value in item.values())
                    if has_numeric:
                        line_items.append(item)
            
            return line_items
    
    # Fallback method: look for patterns in the text that indicate line items
    line_items = []
    price_pattern = r'[\d,]+\.\d{2}'
    quantity_pattern = r'^\d+$'
    
    # Group items that are close to each other horizontally
    potential_line_items = []
    
    # Find lines that have a description of items (looking for product names and descriptions)
    product_rows = []
    for idx, (y, row) in enumerate(sorted_rows):
        # Skip if it looks like a header, total row, or contains non-line item keywords
        row_text = " ".join([item['text'].lower() for item in row])
        non_line_item_keywords = header_keywords + ['subtotal', 'total', 'tax', 'balance', 'terms', 'conditions', 'notes', 'gss']
        if any(word in row_text for word in non_line_item_keywords):
            continue
        
        # Check if row has price-like content and quantity-like content
        has_price = any(re.search(price_pattern, item['text']) for item in row)
        has_quantity = any(re.match(quantity_pattern, item['text'].strip()) for item in row)
        
        # Look for rows that might contain product descriptions
        # These often have longer text items not matching price or quantity patterns
        has_description = any(len(item['text']) > 10 for item in row)
        
        # If row has required elements for a line item
        if has_price and (has_quantity or has_description):
            # Sort items by X position
            sorted_items = sorted(row, key=lambda item: item['x'])
            
            item = {}
            # Try to identify parts based on content
            for i, text_item in enumerate(sorted_items):
                text = text_item['text']
                
                # If it looks like a quantity (first column)
                if i == 0 and re.match(quantity_pattern, text.strip()):
                    item['quantity'] = text
                # If it looks like a price
                elif re.search(price_pattern, text):
                    if 'price' not in item:
                        item['price'] = text
                    else:
                        item['amount'] = text
                # If it's in the middle columns, likely description
                elif 0 < i < len(sorted_items) - 2:
                    if 'description' not in item:
                        item['description'] = text
                    else:
                        item['description'] += ' ' + text
                # Otherwise process based on position
                elif 'description' not in item:
                    item['description'] = text
                else:
                    item['description'] += ' ' + text
            
            if item:
                # Make sure the item doesn't contain non-line item text
                item_text = ' '.join(item.values()).lower()
                non_line_keywords = ['terms', 'conditions', 'notes', 'subtotal', 'total', 'balance', 'gss']
                if not any(keyword in item_text for keyword in non_line_keywords):
                    potential_line_items.append(item)
    
    return potential_line_items