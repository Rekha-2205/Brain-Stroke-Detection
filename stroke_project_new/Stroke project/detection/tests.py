from django.test import TestCase
from django.contrib.auth import get_user_model
from .models import CTScanImage

User = get_user_model()

class CTScanImageTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            role='physician'
        )
    
    def test_create_ct_scan(self):
        ct_scan = CTScanImage.objects.create(
            uploaded_by=self.user,
            patient_id='PT001',
            patient_name='Test Patient',
            patient_age=45,
            patient_gender='Male'
        )
        self.assertEqual(ct_scan.patient_id, 'PT001')
