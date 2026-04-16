"""Simple direct training test"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.training_handler import train_model_background
from admin_panel.models import ModelConfiguration, Dataset

config = ModelConfiguration.objects.first()
dataset = Dataset.objects.get(id=2)

print(f"Starting training directly...")
print(f"Config: {config.config_name} (epochs={config.epochs})")
print(f"Dataset: {dataset.name}")

# Call training function directly (blocks until complete)
train_model_background(config.id, dataset.id, 1)

print(f"\nTraining completed!")
