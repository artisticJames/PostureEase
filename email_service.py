import smtplib
import ssl
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import secrets
import string
from datetime import datetime, timedelta
from config import Config
import logging
import json
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_server = Config.SMTP_SERVER
        self.smtp_port = Config.SMTP_PORT
        self.smtp_username = Config.SMTP_USERNAME
        self.smtp_password = Config.SMTP_PASSWORD
        self.from_email = Config.FROM_EMAIL
        self.base_url = Config.BASE_URL
        # Project logo path (used for CID embedding when BASE_URL is not public)
        self.logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images', 'logo-test-2-1-10.png')
        # EmailJS
        self.emailjs_service_id = Config.EMAILJS_SERVICE_ID
        self.emailjs_public_key = Config.EMAILJS_PUBLIC_KEY
        self.emailjs_private_key = Config.EMAILJS_PRIVATE_KEY
        self.emailjs_template_verify = Config.EMAILJS_TEMPLATE_ID_VERIFY
        self.emailjs_template_reset = Config.EMAILJS_TEMPLATE_ID_RESET
        self.last_error = None

    def _attach_logo_if_needed(self, msg: MIMEMultipart) -> str:
        """Return an <img> tag for the logo.

        - If BASE_URL looks public (not localhost/127.*), prefer absolute URL so many clients cache correctly
        - Otherwise embed the image via CID so it shows even without public hosting
        """
        public_base = (self.base_url and not self.base_url.startswith('http://127.')
                       and not self.base_url.startswith('http://localhost')
                       and not self.base_url.startswith('https://127.')
                       and not self.base_url.startswith('https://localhost'))

        if public_base:
            logo_url = f"{self.base_url}/static/images/logo-test-2-1-10.png"
            return f'<img src="{logo_url}" width="48" height="48" alt="PosturEase" style="border:0;display:block;border-radius:12px;" />'
        else:
            try:
                if os.path.exists(self.logo_path):
                    with open(self.logo_path, 'rb') as f:
                        img = MIMEImage(f.read())
                        img.add_header('Content-ID', '<posturease-logo>')
                        img.add_header('Content-Disposition', 'inline', filename='logo.png')
                        msg.attach(img)
                    return '<img src="cid:posturease-logo" width="48" height="48" alt="PosturEase" style="border:0;display:block;border-radius:12px;" />'
            except Exception:
                # Fallback to text if embed fails
                pass
            # Minimal fallback (no image)
            return '<div style="font:600 18px/24px -apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,Arial,sans-serif;">PosturEase</div>'
        
    def generate_verification_token(self, length=6):
        """Generate a numeric verification code (default 6 digits)."""
        digits = string.digits
        return ''.join(secrets.choice(digits) for _ in range(max(4, length)))
    
    def send_verification_email(self, to_email, username, verification_token, email_type="registration"):
        """Send verification email for registration or password change"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            
            if email_type == "registration":
                msg['Subject'] = "PosturEase - Verify Your Email Address"
                verification_url = f"{Config.BASE_URL}/verify-email?token={verification_token}"
                logo_html = self._attach_logo_if_needed(msg)
                html_content = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <div style="text-align: left; margin-bottom: 16px;">
                            {logo_html}
                        </div>
                        
                        <div style="background-color: #f9f9f9; padding: 30px; border-radius: 10px;">
                            <h2 style="color: #007AFF; margin-bottom: 20px;">Welcome to PosturEase!</h2>
                            
                            <p>Hi {username},</p>
                            
                            <p>Thank you for registering with PosturEase! To complete your registration, please verify your email address using the verification code below:</p>
                            
                            <div style="background-color: #f0f8ff; border: 2px solid #007AFF; border-radius: 8px; padding: 20px; margin: 20px 0; text-align: center;">
                                <h3 style="color: #007AFF; margin: 0 0 10px 0;">Your verification code:</h3>
                                <div style="font-size: 32px; font-weight: bold; color: #007AFF; letter-spacing: 4px; font-family: monospace; background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #ddd;">
                                    {verification_token}
                                </div>
                            </div>
                            
                            <p>Enter this code in the app to verify your email address.</p>
                            
                            <p>Alternatively, you can click the button below to verify automatically:</p>
                            
                            <div style="text-align: center; margin: 30px 0;">
                                <a href="{verification_url}" 
                                   style="background-color: #007AFF; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block;">
                                    Verify Email Address
                                </a>
                            </div>
                            
                            <p>This verification code will expire in 24 hours.</p>
                            
                            <p>If you didn't create an account with PosturEase, please ignore this email.</p>
                            
                            <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
                            <p style="font-size: 12px; color: #666;">
                                This is an automated email from PosturEase. Please do not reply to this email.
                            </p>
                        </div>
                    </div>
                </body>
                </html>
                """
                
            else:  # password change
                msg['Subject'] = "PosturEase - Password Change Verification"
                verification_url = f"{Config.BASE_URL}/verify-password-change?token={verification_token}"
                logo_html = self._attach_logo_if_needed(msg)
                html_content = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <div style="text-align: left; margin-bottom: 16px;">
                            {logo_html}
                        </div>
                        
                        <div style="background-color: #f9f9f9; padding: 30px; border-radius: 10px;">
                            <h2 style="color: #007AFF; margin-bottom: 20px;">Password Change Request</h2>
                            
                            <p>Hi {username},</p>
                            
                            <p>We received a request to change your password. To proceed with the password change, please use the verification code below:</p>
                            
                            <div style="background-color: #f0f8ff; border: 2px solid #007AFF; border-radius: 8px; padding: 20px; margin: 20px 0; text-align: center;">
                                <h3 style="color: #007AFF; margin: 0 0 10px 0;">Your verification code:</h3>
                                <div style="font-size: 32px; font-weight: bold; color: #007AFF; letter-spacing: 4px; font-family: monospace; background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #ddd;">
                                    {verification_token}
                                </div>
                            </div>
                            
                            <p>Enter this code in the app to proceed with the password change.</p>
                            
                            <p>Alternatively, you can click the button below to verify automatically:</p>
                            
                            <div style="text-align: center; margin: 30px 0;">
                                <a href="{verification_url}" 
                                   style="background-color: #007AFF; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block;">
                                    Change Password
                                </a>
                            </div>
                            
                            <p>This verification code will expire in 1 hour.</p>
                            
                            <p>If you didn't request a password change, please ignore this email and your password will remain unchanged.</p>
                            
                            <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
                            <p style="font-size: 12px; color: #666;">
                                This is an automated email from PosturEase. Please do not reply to this email.
                            </p>
                        </div>
                    </div>
                </body>
                </html>
                """
            
            # Attach HTML content
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Verification email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending verification email to {to_email}: {e}")
            return False
    
    def send_password_change_notification(self, to_email, username):
        """Send notification email when password is successfully changed"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = "PosturEase - Password Changed Successfully"
            logo_html = self._attach_logo_if_needed(msg)
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="text-align: left; margin-bottom: 16px;">
                        {logo_html}
                    </div>
                    
                    <div style="background-color: #f9f9f9; padding: 30px; border-radius: 10px;">
                        <h2 style="color: #34C759; margin-bottom: 20px;">Password Changed Successfully</h2>
                        
                        <p>Hi {username},</p>
                        
                        <p>Your PosturEase password has been successfully changed.</p>
                        
                        <p>If you did not make this change, please contact our support team immediately.</p>
                        
                        <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
                        <p style="font-size: 12px; color: #666;">
                            This is an automated email from PosturEase. Please do not reply to this email.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_content, 'html'))
            
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Password change notification sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending password change notification to {to_email}: {e}")
            return False

    def _send_smtp_email(self, to_email: str, subject: str, html_content: str, text_content: str = None) -> bool:
        """Send email via SMTP with HTML and text content"""
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add text content if provided
            if text_content:
                msg.attach(MIMEText(text_content, 'plain'))
            
            # Add HTML content
            msg.attach(MIMEText(html_content, 'html'))
            
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email to {to_email}: {e}")
            self.last_error = str(e)
            return False

    # ---------------- EmailJS (server-side) -----------------
    def _emailjs_send(self, template_id: str, template_params: dict) -> bool:
        if not (self.emailjs_service_id and self.emailjs_private_key and template_id):
            self.last_error = "EmailJS configuration missing. Set EMAILJS_* env vars."
            logger.error(self.last_error)
            return False
        payload = {
            "service_id": self.emailjs_service_id,
            "template_id": template_id,
            "user_id": self.emailjs_public_key or "",
            "accessToken": self.emailjs_private_key,
            "template_params": template_params,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url="https://api.emailjs.com/api/v1.0/email/send",
            data=data,
            headers={
                "Content-Type": "application/json",
                # EmailJS may require a browser-like Origin; supply BASE_URL
                "origin": self.base_url or "http://localhost",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if 200 <= resp.status < 300:
                    self.last_error = None
                    return True
                self.last_error = f"EmailJS send failed: HTTP {resp.status}"
                logger.error(self.last_error)
                return False
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', 'ignore')
            self.last_error = f"EmailJS HTTPError {e.code}: {body}"
            logger.error(self.last_error)
            return False
        except Exception as e:
            self.last_error = f"EmailJS error: {e}"
            logger.error(self.last_error)
            return False

    def send_verification_email_emailjs(self, to_email: str, username: str, verification_token: str) -> bool:
        """Send verification email via EmailJS REST API."""
        logo_url = f"{self.base_url}/static/images/logo-test-2-1-10.png"
        verify_url = f"{self.base_url}/verify-email?token={verification_token}"
        # Provide both styles of variables so either template mapping works
        # Preferred: username, base_url, token, code, logo_url, support_email, year
        # Legacy/Alt: verify_url, to_email, to_name
        params = {
            "to_email": to_email,  # Changed back to to_email to match template
            "to_name": username,
            "username": username,
            "base_url": self.base_url,
            "token": verification_token,
            "code": verification_token,
            "logo_url": logo_url,
            "support_email": self.from_email,
            "year": str(datetime.now().year),
            "verify_url": verify_url,
        }
        return self._emailjs_send(self.emailjs_template_verify, params)

    def send_password_reset_email_emailjs(self, to_email: str, username: str, reset_token: str) -> bool:
        """Send password reset email via EmailJS REST API."""
        logo_url = f"{self.base_url}/static/images/logo-test-2-1-10.png"
        reset_url = f"{self.base_url}/reset-password?token={reset_token}"
        # Provide both styles of variables so either template mapping works
        params = {
            "to_email": to_email,
            "to_name": username,
            "username": username,
            "base_url": self.base_url,
            "token": reset_token,
            "code": reset_token,
            "logo_url": logo_url,
            "support_email": self.from_email,
            "year": str(datetime.now().year),
            "reset_url": reset_url,
        }
        return self._emailjs_send(self.emailjs_template_reset, params)

    def send_otp_email(self, to_email: str, otp_code: str, verification_type: str = "registration") -> bool:
        """Send OTP verification email via EmailJS REST API."""
        logo_url = f"{self.base_url}/static/images/logo-test-2-1-10.png"
        
        # Determine email content based on verification type
        if verification_type == "registration":
            subject = "Complete Your PosturEase Registration"
            message_type = "registration"
        elif verification_type == "password_change":
            subject = "Verify Your Password Change"
            message_type = "password_change"
        else:
            subject = "Verify Your Email"
            message_type = "general"
        
        params = {
            "to_email": to_email,
            "to_name": to_email.split('@')[0],  # Use email prefix as name
            "username": to_email.split('@')[0],
            "base_url": self.base_url,
            "token": otp_code,  # Use 'token' parameter like regular verification
            "code": otp_code,    # Use 'code' parameter for template compatibility
            "logo_url": logo_url,
            "support_email": self.from_email,
            "year": str(datetime.now().year),
        }
        
        # Use the verification template for OTP emails
        return self._emailjs_send(self.emailjs_template_verify, params)

# Create global email service instance
email_service = EmailService()
