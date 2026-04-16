#!/usr/bin/env python
"""Check database content"""
import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.models import ModelConfiguration, Dataset, ModelPerformance

print("Current Data in Database:")
print(f"Configurations: {ModelConfiguration.objects.count()}")
for cfg in ModelConfiguration.objects.all():
    print(f"  - ID:{cfg.id} {cfg.config_name}")

print(f"\nDatasets: {Dataset.objects.count()}")
for ds in Dataset.objects.all():
    print(f"  - ID:{ds.id} {ds.name}")

print(f"\nModelPerformance records: {ModelPerformance.objects.count()}")
