from django.db import models
from django.conf import settings

class Dataset(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    upload_date = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    total_images = models.IntegerField(default=0)
    stroke_images = models.IntegerField(default=0)
    non_stroke_images = models.IntegerField(default=0)
    file_path = models.CharField(max_length=500)
    
    def __str__(self):
        return self.name

class ModelConfiguration(models.Model):
    config_name = models.CharField(max_length=200)
    population_size = models.IntegerField(default=50)
    mutation_rate = models.FloatField(default=0.1)
    crossover_rate = models.FloatField(default=0.8)
    num_generations = models.IntegerField(default=100)
    bilstm_units = models.IntegerField(default=128)
    dropout_rate = models.FloatField(default=0.3)
    learning_rate = models.FloatField(default=0.001)
    batch_size = models.IntegerField(default=32)
    epochs = models.IntegerField(default=50)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)
    
    def __str__(self):
        return self.config_name

class ModelPerformance(models.Model):
    model_version = models.CharField(max_length=50)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    auc_score = models.FloatField()
    training_date = models.DateTimeField(auto_now_add=True)
    configuration = models.ForeignKey(ModelConfiguration, on_delete=models.SET_NULL, null=True)
    dataset_name = models.CharField(max_length=255, default="Unknown", blank=True)
    # store per‑epoch training/validation history (loss/accuracy/etc)
    history = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.model_version} - Accuracy: {self.accuracy}"

class SystemLog(models.Model):
    LOG_TYPES = (
        ('upload', 'Image Upload'),
        ('prediction', 'Model Prediction'),
        ('training', 'Model Training'),
        ('error', 'System Error'),
        ('user_action', 'User Action'),
    )
    log_type = models.CharField(max_length=20, choices=LOG_TYPES)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    description = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.log_type} - {self.timestamp}"
