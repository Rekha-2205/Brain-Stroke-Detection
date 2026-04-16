#!/usr/bin/env python
"""
Run training and don't wait - just start it
"""
import os, sys, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.training_handler import start_model_training
from admin_panel.models import ModelConfiguration, Dataset

config = ModelConfiguration.objects.first()
dataset = Dataset.objects.first()

print(f"Training started with Config {config.id}, Dataset {dataset.id}")
success, msg = start_model_training(config.id, dataset.id, 1)
print(f"Result: {success} - {msg}")
print("Training is running in background thread. Check results in 100+ seconds.")
