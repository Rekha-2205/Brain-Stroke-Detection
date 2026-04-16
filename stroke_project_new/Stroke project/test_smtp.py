import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Get credentials from environment
gmail = os.environ.get('EMAIL_HOST_USER')
password = os.environ.get('EMAIL_HOST_PASSWORD')
recipient = os.environ.get('TEST_EMAIL', 'test@gmail.com')

if not gmail or not password:
    print("ERROR: Set EMAIL_HOST_USER and EMAIL_HOST_PASSWORD env vars first!")
    print("\nExample:")
    print('$env:EMAIL_HOST_USER = "your@gmail.com"')
    print('$env:EMAIL_HOST_PASSWORD = "your-password"')
    print('$env:TEST_EMAIL = "recipient@gmail.com"')
    print('python test_smtp.py')
    exit(1)

print(f"Testing SMTP with:")
print(f"  From: {gmail}")
print(f"  To: {recipient}")
print(f"  Attempting connection to smtp.gmail.com:587...")

try:
    # Create SMTP connection
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    print("✓ TLS connection established")
    
    # Try to login
    server.login(gmail, password)
    print("✓ Login successful")
    
    # Send test email
    msg = MIMEMultipart()
    msg['From'] = gmail
    msg['To'] = recipient
    msg['Subject'] = "Test Email from Stroke Detection Project"
    
    body = "If you see this, SMTP is working correctly and emails should be sent by Django."
    msg.attach(MIMEText(body, 'plain'))
    
    server.send_message(msg)
    print("✓ Email sent successfully!")
    
    server.quit()
    print("\n✓ SMTP is working! Emails should now be delivered by your Django app.")
    
except smtplib.SMTPAuthenticationError:
    print("\n✗ AUTHENTICATION FAILED")
    print("  - Check your Gmail address is correct")
    print("  - Check your password/app-password is correct")
    print("  - If using Gmail, enable 'Less secure app access': https://myaccount.google.com/lesssecureapps")
    
except smtplib.SMTPException as e:
    print(f"\n✗ SMTP ERROR: {e}")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
