import os
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import torch
from utils import preprocess_image, extract_date, extract_invoice_number, extract_line_items

def main():
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    # Initialize YOLOv8 model
    try:
        # Try to load a trained model
        model = YOLO('models/invoice_model.pt')
        print("Loaded custom trained model")
    except Exception as e:
        # If no trained model, use pretrained YOLOv8 model
        print(f"Error loading custom model: {e}")
        model = YOLO('yolov8m.pt')
        print("Using pretrained YOLOv8 model")
    
    # Define the full path to the sample invoice
    image_path = 'data/sample_invoices/Invoice--4--1_jpg.rf.717cd36ba7f4348acc24d2ad3a71b76e.jpg'
    
    # Check if the directory exists, if not create it
    os.makedirs('data/sample_invoices', exist_ok=True)
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Sample invoice not found at {image_path}")
        
        # Print the current working directory for debugging
        print(f"Current working directory: {os.getcwd()}")
        
        # List all files in the data directory (if it exists)
        if os.path.exists('data'):
            print("Files in data directory:")
            for root, dirs, files in os.walk('data'):
                for file in files:
                    print(os.path.join(root, file))
        
        # Try to find an image in the dataset
        found_image = False
        for root, _, files in os.walk('data/yolo_dataset/images'):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    found_image = True
                    break
            if found_image:
                break
    
    print(f"Processing image: {image_path}")
    preprocessed_img, original_img = preprocess_image(image_path)
    
    # Step 1: Use YOLOv8 for field detection
    yolo_results = model(original_img)
    result_image = original_img.copy()
    
    # Extract information from detected fields
    extracted_info = {
        'invoice_number': None,
        'invoice_date': None,
        'line_items': []
    }
    
    # Process YOLOv8 results
    for result in yolo_results:
        boxes = result.boxes.cpu().numpy()
        
        for i, box in enumerate(boxes):
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf = float(box.conf[0])
            
            # Draw bounding box with distinct colors based on field type
            color = (0, 255, 0)  # Default green
            if 'INV' in cls_name:
                color = (255, 0, 0)  # Blue for invoice fields
            elif 'DATE' in cls_name:
                color = (0, 0, 255)  # Red for date fields
            elif 'TOTAL' in cls_name:
                color = (255, 255, 0)  # Cyan for total fields
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_image, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Extract text from this region
            roi = original_img[y1:y2, x1:x2]
            if roi.size > 0:  # Check if ROI is not empty
                ocr_results = reader.readtext(roi)
                extracted_text = " ".join([r[1] for r in ocr_results])
                
                # Store information based on field type
                if cls_name in ['INV_ID', 'INV']:
                    extracted_info['invoice_number'] = extracted_text
                elif cls_name in ['INV_DATE']:
                    extracted_info['invoice_date'] = extracted_text
    
    # Use OCR on the whole image to capture information not detected by YOLOv8
    ocr_results = reader.readtext(original_img)
    texts = [result[1] for result in ocr_results]
    boxes = [result[0] for result in ocr_results]
    full_text = " ".join(texts)
    
    # Extract invoice number and date if not already found
    if not extracted_info['invoice_number']:
        extracted_info['invoice_number'] = extract_invoice_number(full_text)
    
    if not extracted_info['invoice_date']:
        extracted_info['invoice_date'] = extract_date(full_text)
    
    # Extract line items using the enhanced function
    line_items = extract_line_items(original_img, reader, boxes, texts)
    if line_items:
        extracted_info['line_items'] = line_items
    
    # Print results
    print("\n===== EXTRACTED INFORMATION =====")
    print(f"Invoice Number: {extracted_info['invoice_number']}")
    print(f"Invoice Date: {extracted_info['invoice_date']}")
    
    if extracted_info['line_items']:
        print("\nLine Items:")
        for i, item in enumerate(extracted_info['line_items']):
            print(f"Item {i+1}:")
            for key, value in item.items():
                print(f"  {key.capitalize()}: {value}")
    else:
        print("\nNo line items detected")
    
    # Save results with bounding boxes
    cv2.imwrite('invoice_result.jpg', result_image)
    print("Results saved to invoice_result.jpg")

if __name__ == "__main__":
    main()