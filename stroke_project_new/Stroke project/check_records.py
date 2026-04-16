#!/usr/bin/env python
import os, django
os.environ['DJANGO_SETTINGS_MODULE']='stroke_detection.settings'
django.setup()

from admin_panel.models import ModelPerformance

print(f"Total ModelPerformance records: {ModelPerformance.objects.count()}\n")
print("Recent records:")
print("-" * 100)

for p in ModelPerformance.objects.all().order_by('-training_date')[:5]:
    config_name = p.configuration.config_name if p.configuration else "N/A"
    dataset_name = p.dataset_name if p.dataset_name else "N/A"
    print(f"  Version: {p.model_version}")
    print(f"    Config: {config_name}")
    print(f"    Dataset: {dataset_name} ✅" if dataset_name != "N/A" else f"    Dataset: {dataset_name} ❌")
    print(f"    Accuracy: {p.accuracy:.2f}%")
    print(f"    Date: {p.training_date}")
    print()
