from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from django.views.generic import ListView, DetailView
from django.contrib import messages
from django.conf import settings
import os
import time
import numpy as np

from tensorflow.keras.models import load_model

from .models import CTScanImage, StrokeDetectionResult
from .forms import CTScanUploadForm
from admin_panel.models import SystemLog
from ml_models.preprocessing import preprocess_ct_image, save_preprocessed_image
from ml_models.bilstm_model import BiLSTMStrokeDetector


class DashboardView(LoginRequiredMixin, View):
    template_name = 'detection/dashboard.html'

    def get(self, request):
        recent_scans = CTScanImage.objects.filter(
            uploaded_by=request.user
        ).order_by('-upload_date')[:5]

        total_scans = CTScanImage.objects.filter(
            uploaded_by=request.user
        ).count()

        return render(request, self.template_name, {
            'recent_scans': recent_scans,
            'total_scans': total_scans
        })


class CTScanUploadView(LoginRequiredMixin, View):
    template_name = 'detection/upload.html'

    def get(self, request):
        return render(request, self.template_name, {
            'form': CTScanUploadForm()
        })

    def post(self, request):
        form = CTScanUploadForm(request.POST, request.FILES)
        if form.is_valid():
            ct_scan = form.save(commit=False)
            ct_scan.uploaded_by = request.user
            ct_scan.save()

            SystemLog.objects.create(
                log_type='upload',
                user=request.user,
                description=f'CT scan uploaded for patient {ct_scan.patient_name}'
            )

            messages.success(request, 'CT scan uploaded successfully!')
            return redirect('detection:process', pk=ct_scan.pk)

        return render(request, self.template_name, {'form': form})


class CTScanProcessView(LoginRequiredMixin, View):
    template_name = 'detection/processing.html'

    def get(self, request, pk):
        ct_scan = get_object_or_404(
            CTScanImage,
            pk=pk,
            uploaded_by=request.user
        )

        try:
            start_time = time.time()

            # ---------- PREPROCESS ----------
            original_path = ct_scan.original_image.path
            preprocessed_array = preprocess_ct_image(original_path)

            preprocessed_dir = os.path.join(
                settings.MEDIA_ROOT,
                'ct_scans/preprocessed'
            )
            os.makedirs(preprocessed_dir, exist_ok=True)

            filename = f'preprocessed_{ct_scan.id}.png'
            save_path = os.path.join(preprocessed_dir, filename)
            save_preprocessed_image(preprocessed_array, save_path)

            ct_scan.preprocessed_image = f'ct_scans/preprocessed/{filename}'
            ct_scan.save()

            # ---------- LOAD MODEL ----------
            model_path = os.path.join(
                settings.BASE_DIR,
                'ml_models/trained_models/bilstm_model.h5'
            )

            model = load_model(model_path)

            # ---------- PREDICTION ----------
            image_input = np.expand_dims(preprocessed_array, axis=-1)
            image_input = np.expand_dims(image_input, axis=0)

            stroke_prob = float(model.predict(image_input, verbose=0)[0][0])

            if stroke_prob >= 0.5:
                prediction = 'stroke'
                confidence_score = stroke_prob * 100
            else:
                prediction = 'non-stroke'
                confidence_score = (1 - stroke_prob) * 100

            processing_time = time.time() - start_time

            result = StrokeDetectionResult.objects.create(
                ct_scan=ct_scan,
                prediction=prediction,
                confidence_score=round(confidence_score, 2),
                processing_time=processing_time,
                model_version='v1.0'
            )

            SystemLog.objects.create(
                log_type='prediction',
                user=request.user,
                description=(
                    f'Prediction for {ct_scan.patient_name}: '
                    f'{prediction} ({confidence_score:.2f}%)'
                )
            )

            messages.success(request, 'CT scan processed successfully!')
            return redirect('detection:result', pk=result.pk)

        except Exception as e:
            SystemLog.objects.create(
                log_type='error',
                user=request.user,
                description=f'Error processing CT scan: {str(e)}'
            )
            messages.error(request, f'Error: {str(e)}')
            return redirect('detection:dashboard')


class DetectionResultView(LoginRequiredMixin, DetailView):
    model = StrokeDetectionResult
    template_name = 'detection/result.html'
    context_object_name = 'result'

    def get_queryset(self):
        return StrokeDetectionResult.objects.filter(
            ct_scan__uploaded_by=self.request.user
        )


class CaseHistoryView(LoginRequiredMixin, ListView):
    model = CTScanImage
    template_name = 'detection/case_history.html'
    context_object_name = 'scans'
    paginate_by = 10

    def get_queryset(self):
        return CTScanImage.objects.filter(
            uploaded_by=self.request.user
        ).order_by('-upload_date')
