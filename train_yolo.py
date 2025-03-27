from ultralytics import YOLO
import os
import cv2
import numpy as np
import shutil
from pathlib import Path

def train_yolo_model():
    """Train a YOLOv8 model for invoice field detection"""
    print("Starting YOLOv8 training...")
    
    # Use existing dataset YAML file
    dataset_yaml = Path('data/yolo_dataset/data.yaml')
    
    # Check if dataset and YAML file exist
    if not dataset_yaml.exists():
        print(f"Dataset configuration file not found at {dataset_yaml}")
        print("Please ensure the dataset.yaml file exists in data/yolo_dataset")
        return
    
    print(f"Using existing dataset configuration: {dataset_yaml}")
    
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8m.pt')
    
    # Train the model
    try:
        # For quick demo, we'll use a small number of epochs
        model.train(data=str(dataset_yaml.absolute()), epochs=10, imgsz=640, batch=8)
        print("Training completed!")
        
        # Save the trained model
        os.makedirs('models', exist_ok=True)
        shutil.copy('runs/detect/train4/weights/best.pt', 'models/invoice_model.pt')
        print("Model saved to models/invoice_model.pt")
    except Exception as e:
        print(f"Training failed: {e}")
        print("Using pre-trained model for demo")
        print("Error details:", str(e))

if __name__ == "__main__":
    train_yolo_model()