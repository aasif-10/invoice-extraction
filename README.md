# Invoice Information Extraction System

A computer vision system that extracts structured information from invoice images using OCR and object detection.

## Project Overview

This system extracts key information from invoice images, including:
- Invoice numbers
- Invoice dates
- Line items (products, quantities, prices, amounts)
- And other structured data

The solution combines OCR (Optical Character Recognition) with YOLOv8 object detection to locate and extract information from various invoice formats. The model structure is designed to be scalable, allowing for training on additional fields as needed.

## Features

- **Field Detection**: Uses YOLOv8 to detect invoice fields like invoice numbers, dates, and table regions
- **OCR Integration**: Utilizes EasyOCR for text extraction from detected regions
- **Line Item Extraction**: Advanced table detection and parsing for product information
- **Custom Model Training**: Support for training custom YOLOv8 models on specific invoice formats
- **Preprocessing**: Image enhancement for better OCR results
- **Fallback Methods**: Multiple strategies for extracting data when primary methods fail

## Repository Structure

```
├── extract_invoice_info.py  # Main script for processing invoice images
├── train_yolo.py            # Script for training YOLOv8 model on invoice data
├── utils.py                 # Utility functions for preprocessing and text extraction
├── data/                    # Directory for invoice datasets
│   ├── sample_invoices/     # Sample invoice images
│   └── yolo_dataset/        # Dataset for YOLO training with annotations
├── models/                  # Directory for trained models
└── README.md                # This file
```

## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd invoice-extraction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.7+
- OpenCV (cv2)
- numpy
- easyocr
- ultralytics (YOLOv8)
- torch

## Usage

### Processing an Invoice

To extract information from an invoice image:

```bash
python extract_invoice_info.py
```

By default, this will process the sample invoice included in the `data/sample_invoices` directory. The script will:
1. Preprocess the image for better OCR results
2. Use YOLOv8 for field detection
3. Apply EasyOCR to extract text from detected regions
4. Extract line items using table detection algorithms
5. Output structured invoice data and save a visualization

### Training a Custom Model

To train a YOLOv8 model on your own invoice dataset:

1. Prepare your dataset in YOLO format within the `data/yolo_dataset` directory
2. Create a `data.yaml` configuration file specifying classes and paths
3. Run the training script:

```bash
python train_yolo.py
```

The trained model will be saved to `models/invoice_model.pt`.

## How It Works

### 1. Invoice Field Detection

The system uses YOLOv8 to detect regions of interest such as:
- Invoice number fields (INV_ID, INV)
- Date fields (INV_DATE)
- Total amount fields (TOTAL)
- Table regions for line items

### 2. Text Extraction with OCR

EasyOCR is applied to extract text from:
- Detected fields from YOLOv8
- The entire document as a fallback method

### 3. Line Item Extraction

Line items are extracted using multiple strategies:
- Table structure detection
- Row and column alignment analysis
- Pattern recognition for quantities, descriptions, and prices
- Header row detection

### 4. Information Normalization

Extracted data is normalized using regex patterns to identify:
- Date formats
- Invoice number patterns
- Currency values
- Quantities

## Customization

### Adding New Field Types

To train the model for additional field types:
1. Add annotations for new fields in your YOLO dataset
2. Update the `data.yaml` file to include new classes
3. Re-train the model using `train_yolo.py`
4. Update the field extraction logic in `extract_invoice_info.py` to handle new field types

### Improving Text Extraction

The `preprocess_image` function in `utils.py` can be modified to enhance image quality for specific invoice types:
- Adjust threshold parameters
- Add additional preprocessing steps
- Implement document deskewing