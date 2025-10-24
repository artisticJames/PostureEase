#!/usr/bin/env python3
"""
Test script to verify email configuration and send a test email
"""

import os
from dotenv import load_dotenv
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

def test_email_configuration():
    """Test the email configuration and send a test email"""
    
    # Get email configuration
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', '465'))
    smtp_username = os.getenv('SMTP_USERNAME', 'your-email@gmail.com')
    smtp_password = os.getenv('SMTP_PASSWORD', 'your-app-password')
    from_email = os.getenv('FROM_EMAIL', 'your-email@gmail.com')
    
    print("=== Email Configuration Test ===")
    print(f"SMTP Server: {smtp_server}")
    print(f"SMTP Port: {smtp_port}")
    print(f"Username: {smtp_username}")
    print(f"Password: {'*' * len(smtp_password) if smtp_password != 'your-app-password' else 'NOT SET'}")
    print(f"From Email: {from_email}")
    print()
    
    # Check if using default values
    if smtp_username == 'your-email@gmail.com' or smtp_password == 'your-app-password':
        print("❌ ERROR: You're still using default placeholder values!")
        print("Please create a .env file with your actual Gmail credentials.")
        return False
    
    # Test connection
    try:
        print("Testing SMTP connection...")
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            print("✅ SMTP connection successful")
            
            print("Testing login...")
            server.login(smtp_username, smtp_password)
            print("✅ Login successful")
            
            # Send test email
            test_email = input("Enter your email address to send a test email: ")
            if test_email:
                print("Sending test email...")
                
                msg = MIMEMultipart()
                msg['From'] = from_email
                msg['To'] = test_email
                msg['Subject'] = "PosturEase - Email Configuration Test"
                
                body = """
                <html>
                <body>
                    <h2>PosturEase Email Test</h2>
                    <p>This is a test email to verify your email configuration is working correctly.</p>
                    <p>If you received this email, your Gmail SMTP setup is working!</p>
                </body>
                </html>
                """
                
                msg.attach(MIMEText(body, 'html'))
                server.send_message(msg)
                print("✅ Test email sent successfully!")
                print(f"Check your inbox at: {test_email}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nCommon issues:")
        print("1. Check if you're using an App Password (not your regular Gmail password)")
        print("2. Make sure 2-Factor Authentication is enabled on your Gmail account")
        print("3. Verify your Gmail address is correct")
        print("4. Check if your .env file is in the project root directory")
        return False
    
    return True

if __name__ == "__main__":
    test_email_configuration()
