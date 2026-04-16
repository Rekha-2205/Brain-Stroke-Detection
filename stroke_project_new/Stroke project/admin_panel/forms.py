from django import forms
from django.conf import settings
import os
from .models import Dataset, ModelConfiguration


class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'file_path']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Real Hospital CT Scans'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control', 
                'rows': 3,
                'placeholder': 'Describe your dataset...'
            }),
            'file_path': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'media/datasets/ct_scans'
            }),
        }
        labels = {
            'name': 'Dataset Name',
            'description': 'Description',
            'file_path': 'Dataset Path'
        }
        help_texts = {
            'file_path': 'Relative path from project root (e.g., media/datasets/ct_scans). Must contain "stroke" and "non_stroke" folders.'
        }
    
    def clean_file_path(self):
        """Validate and normalize file path"""
        file_path = self.cleaned_data.get('file_path', '')
        
        # Remove leading/trailing slashes and whitespace
        file_path = file_path.strip().strip('/\\')
        
        # Convert backslashes to forward slashes
        file_path = file_path.replace('\\', '/')
        
        # Remove any double slashes
        while '//' in file_path:
            file_path = file_path.replace('//', '/')
        
        return file_path


class ModelConfigurationForm(forms.ModelForm):
    class Meta:
        model = ModelConfiguration
        fields = ['config_name', 'population_size', 'mutation_rate', 'crossover_rate',
                  'num_generations', 'bilstm_units', 'dropout_rate', 'learning_rate',
                  'batch_size', 'epochs']
        widgets = {
            'config_name': forms.TextInput(attrs={'class': 'form-control'}),
            'population_size': forms.NumberInput(attrs={'class': 'form-control'}),
            'mutation_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'crossover_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'num_generations': forms.NumberInput(attrs={'class': 'form-control'}),
            'bilstm_units': forms.NumberInput(attrs={'class': 'form-control'}),
            'dropout_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'learning_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.0001'}),
            'batch_size': forms.NumberInput(attrs={'class': 'form-control'}),
            'epochs': forms.NumberInput(attrs={'class': 'form-control'}),
        }
