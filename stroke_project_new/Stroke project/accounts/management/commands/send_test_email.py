from django.core.management.base import BaseCommand
from django.contrib.auth.forms import PasswordResetForm


class Command(BaseCommand):
    help = 'Send a password-reset email to the provided address (uses configured email backend)'

    def add_arguments(self, parser):
        parser.add_argument('--email', required=True, help='Recipient email address')
        parser.add_argument('--domain', default='localhost:8000', help='Domain to use in the reset link')
        parser.add_argument('--protocol', default='http', help='Protocol to use in the reset link (http or https)')

    def handle(self, *args, **options):
        email = options['email']
        domain = options['domain']
        protocol = options['protocol']

        form = PasswordResetForm({'email': email})
        if not form.is_valid():
            self.stdout.write(self.style.ERROR(f'No active user found with email: {email}'))
            return

        # Use the form to send the email so it uses the project's templates and settings
        form.save(
            domain_override=domain,
            use_https=(protocol == 'https'),
            from_email=None,
            email_template_name='registration/password_reset_email.html',
        )

        self.stdout.write(self.style.SUCCESS(f'Password reset email queued/sent for {email}'))
