import os
import uuid
from django.core.files.storage import default_storage
from django.conf import settings

def generate_unique_filename(filename):
    """Generate unique filename"""
    ext = filename.split('.')[-1]
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    return unique_name

def save_uploaded_file(uploaded_file, subfolder='uploads'):
    """Save uploaded file and return path"""
    filename = generate_unique_filename(uploaded_file.name)
    filepath = os.path.join(subfolder, filename)
    
    path = default_storage.save(filepath, uploaded_file)
    return path

def delete_file(filepath):
    """Delete file from storage"""
    if default_storage.exists(filepath):
        default_storage.delete(filepath)
        return True
    return False

def get_file_size_mb(filepath):
    """Get file size in MB"""
    if default_storage.exists(filepath):
        size = default_storage.size(filepath)
        return size / (1024 * 1024)
    return 0

def validate_image_file(file):
    """Validate image file"""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.dcm']
    ext = os.path.splitext(file.name)[1].lower()
    
    if ext not in valid_extensions:
        return False, f"Invalid file type. Allowed: {', '.join(valid_extensions)}"
    
    if file.size > 10 * 1024 * 1024:  # 10MB
        return False, "File size must be less than 10MB"
    
    return True, "Valid"
