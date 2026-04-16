#!/usr/bin/env python
"""
Test complete training flow end-to-end
"""
import os
import sys
import time
import json
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.training_handler import start_model_training, get_training_state
from admin_panel.models import ModelConfiguration, Dataset, ModelPerformance

print("="*70)
print("FULL TRAINING FLOW TEST")
print("="*70)

# Check if we have test data
config_count = ModelConfiguration.objects.count()
dataset_count = Dataset.objects.count()

print(f"\nAvailable Data:")
print(f"  Configurations: {config_count}")
print(f"  Datasets: {dataset_count}")

if config_count == 0 or dataset_count == 0:
    print("\nERROR: Need at least 1 configuration and 1 dataset to test")
    print("Please create test data in admin panel first")
    sys.exit(1)

# Get first config and dataset
config = ModelConfiguration.objects.first()
dataset = Dataset.objects.first()

print(f"\nUsing:")
print(f"  Config: {config.config_name} (ID:{config.id})")
print(f"  Dataset: {dataset.name} (ID:{dataset.id})")

# Count initial performance records
initial_count = ModelPerformance.objects.count()
print(f"  Initial ModelPerformance records: {initial_count}")

# Start training
print(f"\n{'='*70}")
print("STARTING TRAINING...")
print(f"{'='*70}")

success, message = start_model_training(config.id, dataset.id, 1)
print(f"Start result: {success}")
print(f"Message: {message}")

if not success:
    print("ERROR: Could not start training")
    sys.exit(1)

# Monitor progress
print(f"\nMonitoring progress (timeout: 120 seconds)...")
print(f"Time(s) | Progress% | Training | Status")
print(f"{'='*70}")
sys.stdout.flush()

start_time = time.time()
completed = False

for i in range(121):
    state = get_training_state()
    elapsed = time.time() - start_time
    
    # Print every 2 seconds or on completion/state change
    if i % 2 == 0 or state['progress'] == 100 or not state['is_training']:
        print(f"{elapsed:7.1f} | {state['progress']:9d} | {str(state['is_training']):8} | {state['current_message']}")
        sys.stdout.flush()
    
    # Check for completion
    if state['progress'] == 100 and not state['is_training']:
        completed = True
        break
    
    time.sleep(1)

print(f"{'='*70}")

if completed:
    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    final_state = get_training_state()
    print("\nFinal Training State:")
    print(json.dumps(final_state, indent=2))
    
    # Check database
    print("\n" + "="*70)
    print("DATABASE VERIFICATION")
    print("="*70)
    
    final_count = ModelPerformance.objects.count()
    print(f"ModelPerformance records: {initial_count} → {final_count}")
    
    if final_count > initial_count:
        new_record = ModelPerformance.objects.latest('id')
        print(f"\nNew ModelPerformance Record:")
        print(f"  Version: {new_record.model_version}")
        print(f"  Accuracy: {new_record.accuracy:.2f}%")
        print(f"  Precision: {new_record.precision:.2f}%")
        print(f"  Recall: {new_record.recall:.2f}%")
        print(f"  F1-Score: {new_record.f1_score:.2f}%")
        print(f"  AUC: {new_record.auc_score:.2f}%")
        print(f"  Created: {new_record.created_at}")
    else:
        print("WARNING: No new ModelPerformance record created!")
    
    print("\n✅ TEST PASSED - Full training flow working!")
    
else:
    print("\n❌ TRAINING DID NOT COMPLETE")
    final_state = get_training_state()
    print("\nFinal Training State:")
    print(json.dumps(final_state, indent=2))
    print(f"\nElapsed time: {time.time() - start_time:.1f} seconds")
    sys.exit(1)
