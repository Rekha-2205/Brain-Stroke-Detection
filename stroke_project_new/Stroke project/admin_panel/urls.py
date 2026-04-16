from django.urls import path
from . import views

app_name = 'admin_panel'

urlpatterns = [
    path('dashboard/', views.AdminDashboardView.as_view(), name='dashboard'),
    path('datasets/', views.DatasetListView.as_view(), name='dataset_list'),
    path('datasets/upload/', views.DatasetUploadView.as_view(), name='dataset_upload'),
    path('datasets/validate-path/', views.ValidateDatasetPathView.as_view(), name='dataset_validate_path'),
    path('datasets/<int:pk>/view/', views.DatasetViewDetailView.as_view(), name='dataset_view'),
    path('datasets/<int:pk>/delete/', views.DatasetDeleteView.as_view(), name='dataset_delete'),
    path('configuration/', views.ModelConfigurationView.as_view(), name='model_configuration'),
    path('training/', views.ModelTrainingView.as_view(), name='model_training'),
    path('training/progress/', views.TrainingProgressView.as_view(), name='training_progress'),
    path('training/stop/', views.StopTrainingView.as_view(), name='training_stop'),
    path('performance/', views.ModelPerformanceView.as_view(), name='model_performance'),
    path('users/', views.UserManagementView.as_view(), name='user_management'),
    path('logs/', views.SystemLogsView.as_view(), name='system_logs'),
]
