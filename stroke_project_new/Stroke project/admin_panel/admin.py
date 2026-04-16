from django.contrib import admin
from .models import Dataset, ModelConfiguration, ModelPerformance, SystemLog

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'total_images', 'uploaded_by', 'upload_date']
    list_filter = ['upload_date']

@admin.register(ModelConfiguration)
class ModelConfigurationAdmin(admin.ModelAdmin):
    list_display = ['config_name', 'is_active', 'created_at']
    list_filter = ['is_active']

@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ['model_version', 'accuracy', 'precision', 'recall', 'training_date']
    list_filter = ['training_date']

@admin.register(SystemLog)
class SystemLogAdmin(admin.ModelAdmin):
    list_display = ['log_type', 'user', 'timestamp']
    list_filter = ['log_type', 'timestamp']
