from django.db import models
from detection.models import StrokeDetectionResult

class DiagnosticReport(models.Model):
    detection_result = models.OneToOneField(StrokeDetectionResult, on_delete=models.CASCADE)
    report_file = models.FileField(upload_to='reports/')
    generated_date = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"Report - {self.detection_result.ct_scan.patient_name}"
