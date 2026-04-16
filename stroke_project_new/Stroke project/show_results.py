#!/usr/bin/env python
import os, django
os.environ['DJANGO_SETTINGS_MODULE']='stroke_detection.settings'
django.setup()

from admin_panel.models import ModelPerformance

perfs = ModelPerformance.objects.all().order_by('-training_date')[:5]
print(f"Total ModelPerformance records: {ModelPerformance.objects.count()}\n")
for p in perfs:
    config_name = p.configuration.config_name if p.configuration else "N/A"
    print(f"{p.model_version} | Acc:{p.accuracy:.2f}% | Config:{config_name} | {p.training_date}")
