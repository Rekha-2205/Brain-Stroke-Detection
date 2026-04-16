"""
Advanced preprocessing for CT stroke detection
Includes artifact removal, multi-scale processing, and enhanced normalization
"""
import cv2
import numpy as np
import os


def preprocess_ct_image_advanced(image_path, target_size=(224, 224)):
    """
    Advanced preprocessing for CT scan with medical imaging best practices
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # 1. Artifact removal - bilateral filtering for edge preservation
    img = cv2.bilateralFilter(img.astype(np.float32), 9, 75, 75)
    
    # 2. Adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_uint8 = (np.clip(img, 0, 255)).astype(np.uint8)
    img = clahe.apply(img_uint8).astype(np.float32)
    
    # 3. Gamma correction for intensity normalization
    img = img / 255.0
    gamma = 1.2  # Adjust gamma for stroke visibility
    img = np.power(img, gamma)
    
    # 4. Advanced denoising (Non-Local Means)
    img_uint8 = (img * 255).astype(np.uint8)
    img = cv2.fastNlMeansDenoising(img_uint8, None, h=10, templateWindowSize=7, searchWindowSize=21)
    img = img.astype(np.float32) / 255.0
    
    # 5. Morphological operations for stroke enhancement
    # Apply morphological closing to enhance stroke regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_morph = cv2.morphologyEx(img_uint8, cv2.MORPH_CLOSE, kernel)
    img_morph = img_morph.astype(np.float32) / 255.0
    
    # Blend original with morphological enhancement
    img = 0.7 * img + 0.3 * img_morph
    
    # 6. Resize to target size
    img = cv2.resize(img, target_size)
    
    # 7. Final normalization with standardization
    mean = np.mean(img)
    std = np.std(img)
    if std > 0:
        img = (img - mean) / (std + 1e-7)
    
    # Clip to reasonable range
    img = np.clip(img, -3, 3)
    
    # Rescale to [0, 1]
    img = (img + 3) / 6.0
    
    return img.astype(np.float32)


def preprocess_batch_advanced(image_paths, target_size=(224, 224)):
    """Process multiple images with advanced preprocessing"""
    images = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            img = preprocess_ct_image_advanced(img_path, target_size)
            images.append(img)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Warning: Could not process {img_path}: {e}")
    
    return np.array(images), valid_paths


def save_preprocessed_image(preprocessed_array, output_path):
    """Save preprocessed image to disk"""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to 0-255 range for saving
    img_array = (np.clip(preprocessed_array, 0, 1) * 255).astype(np.uint8)
    
    success = cv2.imwrite(output_path, img_array)
    
    if not success:
        raise IOError(f"Failed to save preprocessed image to {output_path}")
    
    return output_path
