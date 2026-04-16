import os
from django.core.management.base import BaseCommand
from django.conf import settings
from detection.models import CTScanImage
from ml_models.preprocessing import preprocess_ct_image, save_preprocessed_image

class Command(BaseCommand):
    help = 'Reprocess all CT scans to generate preprocessed images'

    def handle(self, *args, **kwargs):
        scans = CTScanImage.objects.all()
        total = scans.count()
        success = 0
        failed = 0
        
        self.stdout.write(f"\nReprocessing {total} CT scans...\n")
        
        for scan in scans:
            try:
                self.stdout.write(f"Processing scan ID {scan.id} ({scan.patient_name})...")
                
                # Check if original image exists
                if not scan.original_image:
                    self.stdout.write(self.style.ERROR(f"  ✗ No original image"))
                    failed += 1
                    continue
                
                if not os.path.exists(scan.original_image.path):
                    self.stdout.write(self.style.ERROR(f"  ✗ Original image file missing"))
                    failed += 1
                    continue
                
                # Preprocess image
                original_path = scan.original_image.path
                preprocessed_array = preprocess_ct_image(original_path)
                
                # Save preprocessed image
                preprocessed_dir = os.path.join(settings.MEDIA_ROOT, 'ct_scans/preprocessed')
                os.makedirs(preprocessed_dir, exist_ok=True)
                
                preprocessed_filename = f'preprocessed_{scan.id}.png'
                preprocessed_path = os.path.join(preprocessed_dir, preprocessed_filename)
                
                save_preprocessed_image(preprocessed_array, preprocessed_path)
                
                # Update database
                scan.preprocessed_image = f'ct_scans/preprocessed/{preprocessed_filename}'
                scan.save()
                
                self.stdout.write(self.style.SUCCESS(f"  ✓ Success"))
                success += 1
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  ✗ Error: {str(e)}"))
                failed += 1
        
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(self.style.SUCCESS(f"Reprocessing complete!"))
        self.stdout.write(f"  Successful: {success}/{total}")
        self.stdout.write(f"  Failed: {failed}/{total}")
        self.stdout.write(f"{'='*60}\n")
