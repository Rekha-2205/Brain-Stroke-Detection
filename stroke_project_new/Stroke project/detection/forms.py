import os
from django import forms
from .models import CTScanImage

class CTScanUploadForm(forms.ModelForm):
    class Meta:
        model = CTScanImage
        fields = ['patient_id', 'patient_name', 'patient_age', 
                  'patient_gender', 'original_image']
        widgets = {
            'patient_id': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Patient ID'
            }),
            'patient_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Patient Name'
            }),
            'patient_age': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Age'
            }),
            'patient_gender': forms.Select(attrs={
                'class': 'form-control'
            }, choices=[
                ('Male', 'Male'),
                ('Female', 'Female'),
                ('Other', 'Other')
            ]),
            'original_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            })
        }
    
    def clean_original_image(self):
        image = self.cleaned_data.get('original_image')
        if image:
            # Validate file size (max 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError("Image file size must be less than 10MB")
            
            # Validate file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.dcm']
            ext = os.path.splitext(image.name)[1].lower()
            if ext not in valid_extensions:
                raise forms.ValidationError(f"Unsupported file extension. Use: {', '.join(valid_extensions)}")
        
        return image
