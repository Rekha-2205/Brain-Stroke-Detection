"""Quick test of training with minimal epochs"""
import os
import django
import time

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.models import ModelConfiguration, Dataset, ModelPerformance, SystemLog
from admin_panel.training_handler import start_model_training, get_training_state

print(f"{'='*70}")
print("STROKE DETECTION MODEL - TRAINING TEST")
print(f"{'='*70}\n")

# Get config and dataset
config = ModelConfiguration.objects.first()
dataset = Dataset.objects.get(id=2)

print(f"Configuration: {config.config_name}")
print(f"  - Epochs: {config.epochs}")
print(f"  - Batch Size: {config.batch_size}")
print(f"  - BiLSTM Units: {config.bilstm_units}")
print(f"\nDataset: {dataset.name}")
print(f"  - Images: {dataset.total_images}")
print(f"  - Stroke: {dataset.stroke_images}")
print(f"  - Non-Stroke: {dataset.non_stroke_images}")

# Start training
print(f"\n{'-'*70}")
print("STARTING TRAINING...")
print(f"{'-'*70}\n")

success, msg = start_model_training(config.id, dataset.id, 1)
if not success:
    print(f"❌ Failed to start training: {msg}")
    exit(1)

print(f"✅ Training started successfully\n")

# Monitor progress
prev_progress = -1
while True:
    state = get_training_state()
    
    # Update display only when progress changes
    if state['progress'] != prev_progress or not state['is_training']:
        print(f"[{state['progress']:3d}%] {state['current_message']}")
        prev_progress = state['progress']
    
    # Check if training completed
    if not state['is_training']:
        print(f"\n{'-'*70}")
        if state['progress'] == 100:
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        else:
            print(f"❌ TRAINING FAILED: {state['current_message']}")
        break
    
    time.sleep(2)

# Display results
print(f"\n{'-'*70}")
print("TRAINING RESULTS")
print(f"{'-'*70}\n")

perf = ModelPerformance.objects.all().order_by('-training_date').first()
if perf:
    print(f"Model Version: {perf.model_version}")
    print(f"Training Date: {perf.training_date}")
    print(f"\nPerformance Metrics:")
    print(f"  - Accuracy:  {perf.accuracy:6.2f}%")
    print(f"  - Precision: {perf.precision:6.2f}%")
    print(f"  - Recall:    {perf.recall:6.2f}%")
    print(f"  - F1-Score:  {perf.f1_score:6.2f}%")
    print(f"  - AUC-ROC:   {perf.auc_score:6.2f}%")
    print(f"\n✅ Model ready for deployment!")
else:
    print("❌ No performance records found")

print(f"\n{'-'*70}")
