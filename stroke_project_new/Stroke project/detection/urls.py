from django.urls import path
from . import views

app_name = 'detection'

urlpatterns = [
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('upload/', views.CTScanUploadView.as_view(), name='upload'),
    path('process/<int:pk>/', views.CTScanProcessView.as_view(), name='process'),
    path('result/<int:pk>/', views.DetectionResultView.as_view(), name='result'),
    path('case-history/', views.CaseHistoryView.as_view(), name='case_history'),
]
