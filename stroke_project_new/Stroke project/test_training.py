"""Test script to verify training functionality"""
import os
import sys
import time
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.training_handler import get_training_state
from admin_panel.models import ModelPerformance, SystemLog

print("\n" + "="*60)
print("TRAINING STATE MONITOR")
print("="*60 + "\n")

for i in range(60):  # Check for up to 60 seconds
    state = get_training_state()
    print(f"[Time: {i}s] Training: {state['is_training']}, Progress: {state['progress']}%, Message: {state['current_message']}")
    
    if not state['is_training'] and state['progress'] == 100:
        print("\n✅ Training completed!")
        break
    
    if not state['is_training'] and state['progress'] == 0:
        print("\n❌ Training failed!")
        break
    
    time.sleep(1)

# Check if model was saved
print("\n" + "="*60)
print("CHECKING RESULTS")
print("="*60 + "\n")

performances = ModelPerformance.objects.all().order_by('-training_date')
if performances:
    latest = performances.first()
    print(f"Latest Model Version: {latest.model_version}")
    print(f"Accuracy: {latest.accuracy:.2f}%")
    print(f"Precision: {latest.precision:.2f}%")
    print(f"Recall: {latest.recall:.2f}%")
    print(f"F1-Score: {latest.f1_score:.2f}%")
    print(f"AUC-ROC: {latest.auc_score:.2f}%")
    print(f"Training Date: {latest.training_date}")
else:
    print("No performance records found yet.")

logs = SystemLog.objects.filter(log_type='training').order_by('-timestamp')
if logs:
    print(f"\nLatest Training Log:")
    print(logs.first().description)
