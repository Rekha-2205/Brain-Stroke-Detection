#!/usr/bin/env python
"""
Test training via HTTP APIs (non-blocking)
"""
import os, django, requests, time, json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from admin_panel.models import ModelConfiguration, Dataset

# Get test data
config = ModelConfiguration.objects.first()
dataset = Dataset.objects.first()

BASE = "http://localhost:8000"

# Start Django server first (run in separate terminal before this)
print("Making sure server is running...")
try:
    resp = requests.get(f"{BASE}/admin-panel/training/", timeout=5)
    print(f"Server check: {resp.status_code}")
except Exception as e:
    print(f"ERROR: Server not running? {e}")
    print("Please start Django server first: python manage.py runserver 0.0.0.0:8000")
    exit(1)

print(f"\nStarting Training:")
print(f"  Config: {config.id} ({config.config_name})")
print(f"  Dataset: {dataset.id} ({dataset.name})")

# Simulate the form submission
data = {
    'configuration': config.id,
    'dataset': dataset.id,
}

print("\nSubmitting training request...")
resp = requests.post(f"{BASE}/admin-panel/training/", data=data)
print(f"Response status: {resp.status_code}")

# Check progress a few times
print("\nChecking progress:")
for i in range(5):
    time.sleep(20)
    resp = requests.get(f"{BASE}/admin-panel/training/progress/")
    if resp.status_code == 200:
        progress = resp.json()
        print(f"  [{i*20}s] Progress: {progress.get('progress', 0)}% Training: {progress.get('is_training', False)}")
    else:
        print(f"  ERROR: {resp.status_code}")

print("\nTraining initiated. It will complete in the background.")
print("Check Performance tab in admin panel to see results.")
