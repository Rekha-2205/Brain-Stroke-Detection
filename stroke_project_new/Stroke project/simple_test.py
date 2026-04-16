#!/usr/bin/env python
"""Simple test: start training, wait for completion, check results"""
import os, sys, time, json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')

# Give server time to start
time.sleep(2)

import django
django.setup()

from admin_panel.training_handler import start_model_training, get_training_state
from admin_panel.models import ModelConfiguration, Dataset, ModelPerformance
from django.db import connections

# Close any stale connections
connections.close_all()

# Get test data
config = ModelConfiguration.objects.first()
dataset = Dataset.objects.first()

if not config or not  dataset:
    print("ERROR: No test data")
    sys.exit(1)

initial_perf_count = ModelPerformance.objects.count()
print(f"Initial ModelPerformance count: {initial_perf_count}")
print(f"Starting training with Config:{config.id} Dataset:{dataset.id}")

# Start training
success, msg = start_model_training(config.id, dataset.id, 1)
if not success:
    print(f"ERROR: {msg}")
    sys.exit(1)

print("Training started. Waiting...")

# Wait for completion with timeout
for sec in range(0, 130, 5):
    time.sleep(5)
    connections.close_all()  # Refresh DB connection
    state = get_training_state()
    
    print(f"[{sec+5:3d}s] Progress:{state['progress']:3d}% Training:{state['is_training']} | {state['current_message'][:40]}")
    
    if state['progress'] == 100 and not state['is_training']:
        print("\n✅ Training completed!")
        break

# Check final results
final_state = get_training_state()
print(f"\nFinal State:\n{json.dumps(final_state, indent=2)}")

# Check database
final_perf_count = ModelPerformance.objects.count()
print(f"\nModelPerformance: {initial_perf_count} → {final_perf_count}")

if final_perf_count > initial_perf_count:
    perf = ModelPerformance.objects.latest('id')
    print(f"\n✅ New record saved successfully!")
    print(f"  Version: {perf.model_version}")
    print(f"  Accuracy: {perf.accuracy:.2f}%")
else:
    print(f"\n⚠️  No new records saved")
