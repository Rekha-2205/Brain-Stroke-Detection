"""
Data augmentation for medical imaging with medical-appropriate techniques
Includes rotations, elastic deformations, and intensity variations
"""
import numpy as np
import cv2
from scipy import ndimage
import albumentations as A


def get_medical_augmentation_pipeline(p=0.7):
    """
    Create augmentation pipeline suitable for medical CT scans
    Avoids unrealistic transformations
    """
    transform = A.Compose([
        # Rotation - moderate angles for anatomical plausibility
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        
        # Elastic deformation - simulates scanning variability
        A.ElasticTransform(alpha=30, sigma=5, p=0.3),
        
        # Grid distortion - subtle spatial variations
        A.GridDistortion(num_steps=4, distort_limit=0.15, p=0.3),
        
        # Intensity augmentation
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        
        # Gaussian noise - simulates imaging noise
        A.GaussNoise(p=0.2),
        
        # Blur - camera motion simulation
        A.Blur(blur_limit=3, p=0.2),
        
        # Morphological operations
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
        
    ], p=p)
    
    return transform


def augment_image(image, transform=None):
    """Apply augmentation to single image"""
    if transform is None:
        transform = get_medical_augmentation_pipeline()
    
    # Convert to 0-255 for albumentations
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Apply transform
    augmented = transform(image=img_uint8)
    
    # Convert back to 0-1
    return augmented['image'].astype(np.float32) / 255.0


def augment_batch(images, labels, augmentation_factor=2, transform=None):
    """
    Augment dataset by creating multiple versions of each image
    
    Args:
        images: array of images
        labels: array of labels
        augmentation_factor: how many augmented versions per image
        transform: albumentations transform pipeline
    """
    if transform is None:
        transform = get_medical_augmentation_pipeline()
    
    augmented_images = [images]
    augmented_labels = [labels]
    
    for _ in range(augmentation_factor - 1):
        aug_batch = np.array([augment_image(img, transform) for img in images])
        augmented_images.append(aug_batch)
        augmented_labels.append(labels)
    
    # Concatenate all augmented data
    all_images = np.concatenate(augmented_images, axis=0)
    all_labels = np.concatenate(augmented_labels, axis=0)
    
    return all_images, all_labels


def augment_batch_online(images, labels, transform=None):
    """
    Online augmentation during training (one pass per epoch)
    More memory efficient than pre-augmentation
    """
    if transform is None:
        transform = get_medical_augmentation_pipeline()
    
    augmented = np.array([augment_image(img, transform) for img in images])
    return augmented, labels
