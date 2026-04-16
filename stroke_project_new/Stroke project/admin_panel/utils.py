from django.db.models import Count, Avg
from detection.models import CTScanImage, StrokeDetectionResult
from accounts.models import CustomUser

def get_system_statistics():
    """Get system-wide statistics"""
    stats = {
        'total_users': CustomUser.objects.count(),
        'total_scans': CTScanImage.objects.count(),
        'total_predictions': StrokeDetectionResult.objects.count(),
        'stroke_detected': StrokeDetectionResult.objects.filter(prediction='stroke').count(),
        'non_stroke_detected': StrokeDetectionResult.objects.filter(prediction='non-stroke').count(),
        'avg_confidence': StrokeDetectionResult.objects.aggregate(Avg('confidence_score'))['confidence_score__avg'],
        'users_by_role': CustomUser.objects.values('role').annotate(count=Count('id')),
    }
    return stats

def get_user_statistics(user):
    """Get user-specific statistics"""
    stats = {
        'total_uploads': CTScanImage.objects.filter(uploaded_by=user).count(),
        'stroke_cases': StrokeDetectionResult.objects.filter(
            ct_scan__uploaded_by=user, 
            prediction='stroke'
        ).count(),
        'non_stroke_cases': StrokeDetectionResult.objects.filter(
            ct_scan__uploaded_by=user, 
            prediction='non-stroke'
        ).count(),
    }
    return stats
