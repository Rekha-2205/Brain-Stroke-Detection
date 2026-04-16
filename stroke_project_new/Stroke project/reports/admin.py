from django.contrib import admin
from .models import DiagnosticReport

@admin.register(DiagnosticReport)
class DiagnosticReportAdmin(admin.ModelAdmin):
    list_display = ['detection_result', 'generated_date']
    list_filter = ['generated_date']
