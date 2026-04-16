import logging
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(MiddlewareMixin):
    """Log all requests"""
    
    def process_request(self, request):
        logger.info(f"{request.method} {request.path} - {request.user}")
        return None

class ExceptionMiddleware(MiddlewareMixin):
    """Handle exceptions globally"""
    
    def process_exception(self, request, exception):
        logger.error(f"Exception: {exception} - Path: {request.path}")
        return None
