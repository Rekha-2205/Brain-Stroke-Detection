"""Test stop training feature"""
import os
import django
import time

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.models import ModelConfiguration, Dataset
from admin_panel.training_handler import start_model_training, get_training_state, request_stop_training

print("\n" + "="*70)
print("TESTING STOP TRAINING FEATURE")
print("="*70 + "\n")

config = ModelConfiguration.objects.first()
dataset = Dataset.objects.get(id=2)

print(f"Config: {config.config_name} (epochs={config.epochs})")
print(f"Dataset: {dataset.name}")
print(f"\nStarting training...")

success, msg = start_model_training(config.id, dataset.id, 1)
print(f"✅ {msg}\n")

# Let it train for a bit
time.sleep(5)

# Request stop
print("Requesting training to stop after 5 seconds...\n")
request_stop_training()

# Monitor until stopped
print("Monitoring training status:\n")
for i in range(120):  # Up to 2 minutes
    state = get_training_state()
    if i % 5 == 0 or not state['is_training']:
        print(f"  [{i}s] Training: {state['is_training']}, Progress: {state['progress']}%, Message: {state['current_message']}")
    
    if not state['is_training']:
        print(f"\n✅ Training stopped/completed!")
        break
    
    time.sleep(1)

print("\n" + "="*70)
