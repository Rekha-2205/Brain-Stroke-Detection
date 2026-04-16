from django.core.management.base import BaseCommand
from detection.models import CTScanImage
import os

class Command(BaseCommand):
    help = 'Check for missing preprocessed images'

    def handle(self, *args, **kwargs):
        scans = CTScanImage.objects.all()
        
        for scan in scans:
            self.stdout.write(f"\nChecking scan ID {scan.id}:")
            
            # Check original
            if scan.original_image:
                if os.path.exists(scan.original_image.path):
                    self.stdout.write(f"  ✓ Original: {scan.original_image.path}")
                else:
                    self.stdout.write(self.style.ERROR(f"  ✗ Original missing: {scan.original_image.path}"))
            
            # Check preprocessed
            if scan.preprocessed_image:
                if os.path.exists(scan.preprocessed_image.path):
                    self.stdout.write(f"  ✓ Preprocessed: {scan.preprocessed_image.path}")
                else:
                    self.stdout.write(self.style.WARNING(f"  ✗ Preprocessed missing: {scan.preprocessed_image.path}"))
            else:
                self.stdout.write(self.style.WARNING(f"  ! No preprocessed image in database"))
