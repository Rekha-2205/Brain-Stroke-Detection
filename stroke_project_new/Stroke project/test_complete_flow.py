#!/usr/bin/env python
"""
Test complete training with dataset name capture and display
1. Start training
2. Wait for completion (1-2 minutes)
3. Verify configuration and dataset are saved and displayable
"""
import os, sys, time, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.training_handler import start_model_training, get_training_state
from admin_panel.models import ModelConfiguration, Dataset, ModelPerformance

print("="*80)
print("COMPLETE TRAINING TEST WITH DATASET CAPTURE")
print("="*80)

# Get test data
config = ModelConfiguration.objects.first()
dataset = Dataset.objects.first()

if not config or not dataset:
    print("ERROR: Missing test data")
    sys.exit(1)

print(f"\n📋 TEST SETUP:")
print(f"   Configuration: ID {config.id}: {config.config_name}")
print(f"   Dataset: ID {dataset.id}: {dataset.name}")
print(f"   Initial ModelPerformance records: {ModelPerformance.objects.count()}")

initial_count = ModelPerformance.objects.count()

# Start training
print(f"\n{'='*80}")
print("🚀 STARTING TRAINING")
print(f"{'='*80}")

success, msg = start_model_training(config.id, dataset.id, 1)

if not success:
    print(f"ERROR: {msg}")
    sys.exit(1)

print(f"✅ Training started in background")
print(f"   Estimated duration: 90-100 seconds")
print(f"   Waiting for completion...\n")

# Wait for training to complete
time.sleep(100)

print(f"\n{'='*80}")
print("CHECKING RESULTS")
print(f"{'='*80}\n")

# Check training state
state = get_training_state()
print(f"Training State:")
print(f"   Progress: {state['progress']}%")
print(f"   Is Training: {state['is_training']}")
print(f"   Message: {state['current_message']}")

# Check database
final_count = ModelPerformance.objects.count()
print(f"\nDatabase Update:")
print(f"   Before: {initial_count} records")
print(f"   After: {final_count} records")

if final_count <= initial_count:
    print(f"\n❌ ERROR: No new training record created!")
    sys.exit(1)

# Get latest result
latest = ModelPerformance.objects.latest('training_date')

print(f"\n📊 LATEST TRAINING RESULT:")
print(f"   Model Version: {latest.model_version}")
print(f"   Accuracy: {latest.accuracy:.2f}%")
print(f"   Precision: {latest.precision:.2f}%")
print(f"   Recall: {latest.recall:.2f}%")
print(f"   F1-Score: {latest.f1_score:.2f}%")
print(f"   AUC-ROC: {latest.auc_score:.2f}%")

print(f"\n🔗 CONFIGURATION & DATASET:")
if latest.configuration:
    print(f"   ✅ Configuration: {latest.configuration.config_name}")
    print(f"      (BiLSTM Units: {latest.configuration.bilstm_units})")
else:
    print(f"   ❌ No configuration linked!")

if latest.dataset_name:
    print(f"   ✅ Dataset: {latest.dataset_name}")
else:
    print(f"   ❌ No dataset name saved!")

print(f"\n{'='*80}")
if latest.configuration and latest.dataset_name:
    print("✅ SUCCESS - Complete flow working!")
    print(f"{'='*80}")
    print(f"\nNow open http://localhost:8000/admin-panel/performance/")
    print(f"You should see:")
    print(f"  - Configuration: {latest.configuration.config_name}")
    print(f"  - Dataset: {latest.dataset_name}")
    print(f"  - All metrics displayed")
else:
    print("❌ FAILED - Missing configuration or dataset!")
    sys.exit(1)
