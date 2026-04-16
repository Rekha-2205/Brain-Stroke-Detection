#!/usr/bin/env python
"""
FINAL TEST - Start Django server, run training, verify completion
"""
import subprocess, time, os, sys, threading

os.chdir(r"c:\vinay\Rekha major project\Stroke project")

# Start Django server in background
print("Starting Django server...")
server_proc = subprocess.Popen(
    [sys.executable, "manage.py", "runserver", "0.0.0.0:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait for server to start
time.sleep(5)

print("Server started. Running training test...")

# Now run the actual training
try:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
    import django
    django.setup()
    
    from admin_panel.training_handler import start_model_training, get_training_state
    from admin_panel.models import ModelConfiguration, Dataset, ModelPerformance
    
    # Check initial state
    initial_count = ModelPerformance.objects.count()
    print(f"Initial ModelPerformance count: {initial_count}")
    
    config = ModelConfiguration.objects.first()
    dataset = Dataset.objects.first()
    
    if not config or not dataset:
        print("ERROR: No test data")
        sys.exit(1)
    
    print(f"\nStarting training with Config:{config.id} Dataset:{dataset.id}")
    success, msg = start_model_training(config.id, dataset.id, 1)
    
    if not success:
        print(f"ERROR: {msg}")
        sys.exit(1)
    
    print("Training started. Monitoring...")
    
    # Wait for completion (max 120 seconds)
    for elapsed in range(0, 121, 5):
        time.sleep(5)
        
        state = get_training_state()
        print(f"[{elapsed+5:3d}s] Progress: {state['progress']:3d}% | Training: {state['is_training']} | {state['current_message']}")
        
        if state['progress'] == 100 and not state['is_training']:
            print("\n✅ Training completed!")
            break
    
    # Check results
    final_count = ModelPerformance.objects.count()
    print(f"\nFinal ModelPerformance count: {initial_count} → {final_count}")
    
    if final_count > initial_count:
        perf = ModelPerformance.objects.latest('id')
        print(f"\n✅ SUCCESS - New record saved!")
        print(f"  Version: {perf.model_version}")
        print(f"  Accuracy: {perf.accuracy:.2f}%")
        print(f"  Precision: {perf.precision:.2f}%")
        print(f"  Recall: {perf.recall:.2f}%")
        print(f"  F1: {perf.f1_score:.2f}%")
        print(f"  AUC: {perf.auc_score:.2f}%")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED - No new record saved!")
        sys.exit(1)

finally:
    print("\nShutting down server...")
    server_proc.terminate()
    server_proc.wait(timeout=5)
