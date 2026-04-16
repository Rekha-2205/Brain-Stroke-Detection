from django.urls import path
from . import views

app_name = 'reports'

urlpatterns = [
    path('generate/<int:pk>/', views.GenerateReportView.as_view(), name='generate'),
]
