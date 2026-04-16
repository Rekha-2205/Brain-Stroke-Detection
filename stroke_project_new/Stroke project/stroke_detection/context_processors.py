from django.conf import settings

def site_info(request):
    """Add site information to all templates"""
    return {
        'SITE_NAME': 'Stroke Detection System',
        'SITE_VERSION': '1.0.0',
        'DEBUG_MODE': settings.DEBUG,
    }

def user_context(request):
    """Add user-specific context"""
    if request.user.is_authenticated:
        return {
            'unread_notifications': 0,  # Add notification logic
            'user_role': request.user.role,
        }
    return {}
