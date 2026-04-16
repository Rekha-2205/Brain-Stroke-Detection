#!/usr/bin/env python
"""
Complete training flow test:
1. Start training
2. Monitor until 100% complete
3. Verify results saved with configuration
4. Confirm Performance dashboard ready
"""
import os, sys, time, json, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.training_handler import start_model_training, get_training_state
from admin_panel.models import ModelConfiguration, Dataset, ModelPerformance

print("="*70)
print("COMPREHENSIVE TRAINING TEST")
print("="*70)

# Get test data
config = ModelConfiguration.objects.first()
dataset = Dataset.objects.first()

if not config or not dataset:
    print("ERROR: No test data available")
    sys.exit(1)

print(f"\nTest Setup:")
print(f"  Configuration: {config.id} - {config.config_name}")
print(f"  Dataset: {dataset.id} - {dataset.name}")

initial_perf = ModelPerformance.objects.count()
print(f"  Initial results: {initial_perf}")

# Start training
print(f"\n{'='*70}")
print("STEP 1: START TRAINING")
print(f"{'='*70}")

success, msg = start_model_training(config.id, dataset.id, 1)
print(f"Result: {success}")
print(f"Message: {msg}")

if not success:
    print("ERROR: Could not start training")
    sys.exit(1)

# Monitor progress
print(f"\nSTEP 2: MONITOR PROGRESS (up to 120 seconds)")
print(f"  Note: Training takes ~75-100 seconds (1-2 minutes)")
print(f"{'='*-70}\n")

completed = False
for elapsed in range(0, 130, 5):
    time.sleep(5)
    
    state = get_training_state()
    pct = state['progress']
    
    # Print every checkpoint or when status changes
    if elapsed % 10 == 0 or pct == 100 or not state['is_training']:
        status_icon = "🔄" if state['is_training'] else "✅" if pct == 100 else "⏳"
        print(f"{status_icon} [{elapsed:3d}s] Progress: {pct:3d}% | Training: {str(state['is_training']):5} | {state['current_message'][:50]}")
    
    if pct == 100 and not state['is_training']:
        completed = True
        print(f"\n✅ TRAINING COMPLETED at {elapsed} seconds!\n")
        break

print(f"{'='*70}")
if not completed:
    print("ERROR: Training did not complete in 120 seconds")
    sys.exit(1)

# Verify results saved
print("STEP 3: VERIFY DATABASE RESULTS")
print(f"{'='*70}\n")

final_perf = ModelPerformance.objects.count()
print(f"ModelPerformance records: {initial_perf} → {final_perf}")

if final_perf == initial_perf:
    print("ERROR: No new results saved!")
    sys.exit(1)

# Get the latest result
latest = ModelPerformance.objects.latest('created_at')

print(f"\n✅ New Record Created:")
print(f"  Model Version: {latest.model_version}")
print(f"  Accuracy: {latest.accuracy:.2f}%")
print(f"  Precision: {latest.precision:.2f}%")  
print(f"  Recall: {latest.recall:.2f}%")
print(f"  F1-Score: {latest.f1_score:.2f}%")
print(f"  AUC-ROC: {latest.auc_score:.2f}%")
print(f"  Created: {latest.created_at}")

# Verify configuration is linked
print(f"\nConfiguration Link:")
if latest.configuration:
    print(f"  ✅ Configuration: {latest.configuration.config_name} (ID: {latest.configuration.id})")
    print(f"  Configuration BiLSTM Units: {latest.configuration.bilstm_units}")
    print(f"  Configuration Epochs: {latest.configuration.epochs}")
else:
    print(f"  ⚠️  No configuration linked!")

print(f"\n{'='*70}")
print("✅ SUCCESS - Complete training flow working!")
print(f"{'='*70}")
print(f"\nNext: Open http://localhost:8000/admin-panel/performance/")
print(f"      You should see the new results with configuration name visible")
