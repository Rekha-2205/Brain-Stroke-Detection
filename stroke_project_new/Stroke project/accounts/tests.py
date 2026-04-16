from django.test import TestCase
from django.contrib.auth import get_user_model

User = get_user_model()

class CustomUserTests(TestCase):
    def test_create_user(self):
        user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123',
            role='physician'
        )
        self.assertEqual(user.username, 'testuser')
        self.assertEqual(user.role, 'physician')
        self.assertTrue(user.is_active)
