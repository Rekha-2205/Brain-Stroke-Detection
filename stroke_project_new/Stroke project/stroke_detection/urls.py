"""
URL configuration for stroke_detection project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # Root redirect
    path('', RedirectView.as_view(url='/accounts/login/', permanent=False), name='home'),
    
    # App URLs
    path('accounts/', include('accounts.urls')),
    path('detection/', include('detection.urls')),
    path('admin-panel/', include('admin_panel.urls')),
    path('reports/', include('reports.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# Custom error handlers
handler404 = 'django.views.defaults.page_not_found'
handler500 = 'django.views.defaults.server_error'
handler403 = 'django.views.defaults.permission_denied'
handler400 = 'django.views.defaults.bad_request'

# Admin site customization
admin.site.site_header = "Stroke Detection System Admin"
admin.site.site_title = "Stroke Detection Admin"
admin.site.index_title = "Welcome to Stroke Detection System Administration"
