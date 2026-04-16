"""Test training with actual model"""
import os
import sys
import django
import time

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.models import ModelConfiguration, Dataset
from admin_panel.training_handler import start_model_training, get_training_state

config = ModelConfiguration.objects.first()
dataset = Dataset.objects.get(id=2)

print(f"Config: {config.config_name} (epochs={config.epochs})")
print(f"Dataset: {dataset.name}")

# Start training
print("\nStarting training...")
success, msg = start_model_training(config.id, dataset.id, 1)
print(f"Result: {success}")
print(f"Message: {msg}")

# Monitor progress
print(f"\n{'='*60}")
print("MONITORING TRAINING PROGRESS")
print(f"{'='*60}\n")

for i in range(600):  # Up to 10 minutes
    state = get_training_state()
    if i % 10 == 0 or state['progress'] == 100:
        print(f"[{i}s] Training: {state['is_training']}, Progress: {state['progress']:3d}%, Message: {state['current_message']}")
    
    if not state['is_training']:
        print(f"\n[FINAL] Training: {state['is_training']}, Progress: {state['progress']}%, Message: {state['current_message']}")
        break
    
    time.sleep(1)
