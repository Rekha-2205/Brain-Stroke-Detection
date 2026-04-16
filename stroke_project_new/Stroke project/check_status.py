#!/usr/bin/env python
"""Check current training status"""
import os, django, json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.training_handler import get_training_state
from admin_panel.models import ModelPerformance

state = get_training_state()
print("Current Training State:")
print(json.dumps(state, indent=2))

count = ModelPerformance.objects.count()
print(f"\nModelPerformance records: {count}")

if count > 0:
    latest = ModelPerformance.objects.latest('id')
    print(f"Latest: {latest.model_version} - Accuracy: {latest.accuracy:.2f}%")
