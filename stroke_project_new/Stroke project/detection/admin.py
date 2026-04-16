from django.contrib import admin
from .models import CTScanImage, StrokeDetectionResult

@admin.register(CTScanImage)
class CTScanImageAdmin(admin.ModelAdmin):
    list_display = ['patient_id', 'patient_name', 'patient_age', 'uploaded_by', 'upload_date']
    list_filter = ['upload_date', 'patient_gender']
    search_fields = ['patient_id', 'patient_name']

@admin.register(StrokeDetectionResult)
class StrokeDetectionResultAdmin(admin.ModelAdmin):
    list_display = ['ct_scan', 'prediction', 'confidence_score', 'prediction_date']
    list_filter = ['prediction', 'prediction_date']
