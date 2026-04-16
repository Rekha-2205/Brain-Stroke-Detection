#!/usr/bin/env python
"""Start new training and wait for completion"""
import os, sys, time, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.training_handler import start_model_training, get_training_state
from admin_panel.models import ModelConfiguration, Dataset, ModelPerformance

config = ModelConfiguration.objects.first()
dataset = Dataset.objects.first()
initial_count = ModelPerformance.objects.count()

print(f"Starting training with:")
print(f"  Config: {config.config_name}")
print(f"  Dataset: {dataset.name}")
print(f"  Current records: {initial_count}")

# Start training
start_model_training(config.id, dataset.id, 1)

# Wait for completion
print(f"Waiting for training (100 seconds)...")
time.sleep(100)

# Check if completed
state = get_training_state()
final_count = ModelPerformance.objects.count()

print(f"\nTraining state: {state['progress']}% - {state['is_training']}")
print(f"Records now: {initial_count} → {final_count}")

if final_count > initial_count:
    latest = ModelPerformance.objects.latest('training_date')
    print(f"\n✅ NEW RECORD CREATED!")
    print(f"Version: {latest.model_version}")
    print(f"Config: {latest.configuration.config_name if latest.configuration else 'N/A'}")
    print(f"Dataset: {latest.dataset_name}")
    print(f"Accuracy: {latest.accuracy:.2f}%")
else:
    print(f"\n⏳ Training still in progress or not yet started")
    print(f"Model state: {state}")
