#!/usr/bin/env python
"""Start training and wait for completion"""
import os, sys, time, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.training_handler import start_model_training, get_training_state
from admin_panel.models import ModelConfiguration, Dataset, ModelPerformance

config = ModelConfiguration.objects.first()
dataset = Dataset.objects.first()
initial = ModelPerformance.objects.count()

print(f"Starting training...")
success, _ = start_model_training(config.id, dataset.id, 1)

if not success:
    print("ERROR: Training not started")
    sys.exit(1)

print(f"Training started. Waiting 130 seconds...")
time.sleep(130)

print(f"Checking results...")
state = get_training_state()
final = ModelPerformance.objects.count()

print(f"\nTraining state: {state['progress']}% Progress - {state['is_training']} Training")
print(f"Results saved: {initial} → {final}")

if final > initial:
    perf = ModelPerformance.objects.latest('created_at')
    print(f"\n✅ SUCCESS!")
    print(f"Accuracy: {perf.accuracy:.2f}%")
    print(f"Config: {perf.configuration.config_name if perf.configuration else 'N/A'}")
else:
    print(f"\n⚠️  No new results yet (still training?)")
