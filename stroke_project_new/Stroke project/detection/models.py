from django.db import models
from django.conf import settings

class CTScanImage(models.Model):
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    patient_id = models.CharField(max_length=100)
    patient_name = models.CharField(max_length=200)
    patient_age = models.IntegerField()
    patient_gender = models.CharField(max_length=10)
    original_image = models.ImageField(upload_to='ct_scans/original/')
    preprocessed_image = models.ImageField(upload_to='ct_scans/preprocessed/', null=True, blank=True)
    upload_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"CT Scan - {self.patient_name} - {self.upload_date}"

class StrokeDetectionResult(models.Model):
    ct_scan = models.OneToOneField(CTScanImage, on_delete=models.CASCADE)
    prediction = models.CharField(max_length=20)  # 'stroke' or 'non-stroke'
    confidence_score = models.FloatField()
    processing_time = models.FloatField()  # in seconds
    prediction_date = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50, default='v1.0')
    
    def __str__(self):
        return f"Result - {self.ct_scan.patient_name} - {self.prediction}"
