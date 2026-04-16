#!/usr/bin/env python
"""Run training in background and save completion status"""
import os, django, time, sys, json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.training_handler import start_model_training, get_training_state
from admin_panel.models import ModelConfiguration, Dataset

config = ModelConfiguration.objects.first()
dataset = Dataset.objects.first()

print(f"Starting training with Config ID:{config.id} Dataset ID:{dataset.id}")
sys.stdout.flush()

success, message = start_model_training(config.id, dataset.id, 1)

if success:
    print("Training started successfully, monitoring for 120 seconds...")
    sys.stdout.flush()
    
    for i in range(120):
        state = get_training_state()
        if i % 10 == 0:
            print(f"[{i}s] Progress: {state['progress']}%")
            sys.stdout.flush()
        
        if state['progress'] == 100 and not state['is_training']:
            print(f"Training completed at {i}s!")
            with open("training_complete.txt", "w") as f:
                f.write(json.dumps(state, indent=2))
            sys.exit(0)
        
        time.sleep(1)
    
    state = get_training_state()
    print(f"Timeout reached. Final state: {state['progress']}% training ={state['is_training']}")
    with open("training_timeout.txt", "w") as f:
        f.write(json.dumps(state, indent=2))
else:
    print(f"ERROR: {message}")
    sys.exit(1)
