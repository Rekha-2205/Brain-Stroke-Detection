"""Debug training function"""
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

try:
    from admin_panel.models import ModelConfiguration, Dataset, SystemLog
    from admin_panel.training_handler import train_model_background
    from django.conf import settings
    
    config = ModelConfiguration.objects.first()
    dataset = Dataset.objects.get(id=2)  # Get the correct dataset
    
    print(f"Config: {config.config_name} (ID: {config.id})")
    print(f"Dataset: {dataset.name} (ID: {dataset.id})")
    
    dataset_path = os.path.join(str(settings.BASE_DIR), dataset.file_path)
    print(f"Dataset path: {dataset_path}")
    print(f"Path exists: {os.path.exists(dataset_path)}")
    
    # Try loading the dataset
    print("\nAttempting to load dataset...")
    from admin_panel.training_handler import load_dataset_from_path
    
    X, y = load_dataset_from_path(dataset_path)
    print(f"Dataset loaded successfully!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Unique labels: {set(y)}")
    
except Exception as e:
    import traceback
    print(f"\nError: {str(e)}")
    print(traceback.format_exc())
    
    # Check logs
    logs = SystemLog.objects.filter(log_type='error').order_by('-timestamp')
    if logs:
        print("\nRecent error logs:")
        for log in logs[:3]:
            print(f"\n{log.timestamp}: {log.description[:500]}")
