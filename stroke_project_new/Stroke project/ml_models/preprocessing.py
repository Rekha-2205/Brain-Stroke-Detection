import cv2
import numpy as np
from PIL import Image
import os

def preprocess_ct_image(image_path, target_size=(224, 224)):
    """
    Preprocess CT scan image for model input
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply((img * 255).astype(np.uint8))
    img = img_clahe.astype(np.float32) / 255.0
    
    # Denoise
    img_uint8 = (img * 255).astype(np.uint8)
    img = cv2.fastNlMeansDenoising(img_uint8, None, 10, 7, 21)
    img = img.astype(np.float32) / 255.0
    
    # ensure image has a channel dimension (Conv2D expects 4D input)
    # original arrays are grayscale so shape is (H, W) – add channel=1
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    return img

def save_preprocessed_image(preprocessed_array, output_path):
    """
    Save preprocessed image to disk
    """
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert normalized array back to 0-255 range
    img_array = (preprocessed_array * 255).astype(np.uint8)
    
    # Save using OpenCV
    success = cv2.imwrite(output_path, img_array)
    
    if not success:
        raise IOError(f"Failed to save preprocessed image to {output_path}")
    
    print(f"✓ Preprocessed image saved successfully to: {output_path}")
    return output_path
