import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
import string
from datetime import datetime, timedelta
from config import Config
import logging

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_server = Config.SMTP_SERVER
        self.smtp_port = Config.SMTP_PORT
        self.smtp_username = Config.SMTP_USERNAME
        self.smtp_password = Config.SMTP_PASSWORD
        self.from_email = Config.FROM_EMAIL
        
    def generate_verification_token(self, length=32):
        """Generate a random verification token"""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
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
                
                html_content = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <div style="text-align: center; margin-bottom: 30px;">
                            <h1 style="color: #007AFF;">PosturEase</h1>
                        </div>
                        
                        <div style="background-color: #f9f9f9; padding: 30px; border-radius: 10px;">
                            <h2 style="color: #007AFF; margin-bottom: 20px;">Welcome to PosturEase!</h2>
                            
                            <p>Hi {username},</p>
                            
                            <p>Thank you for registering with PosturEase! To complete your registration, please verify your email address by clicking the button below:</p>
                            
                            <div style="text-align: center; margin: 30px 0;">
                                <a href="{verification_url}" 
                                   style="background-color: #007AFF; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block;">
                                    Verify Email Address
                                </a>
                            </div>
                            
                            <p>If the button doesn't work, you can copy and paste this link into your browser:</p>
                            <p style="word-break: break-all; color: #666;">{verification_url}</p>
                            
                            <p>This verification link will expire in 24 hours.</p>
                            
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
                
                html_content = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <div style="text-align: center; margin-bottom: 30px;">
                            <h1 style="color: #007AFF;">PosturEase</h1>
                        </div>
                        
                        <div style="background-color: #f9f9f9; padding: 30px; border-radius: 10px;">
                            <h2 style="color: #007AFF; margin-bottom: 20px;">Password Change Request</h2>
                            
                            <p>Hi {username},</p>
                            
                            <p>We received a request to change your password. To proceed with the password change, please click the button below:</p>
                            
                            <div style="text-align: center; margin: 30px 0;">
                                <a href="{verification_url}" 
                                   style="background-color: #007AFF; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block;">
                                    Change Password
                                </a>
                            </div>
                            
                            <p>If the button doesn't work, you can copy and paste this link into your browser:</p>
                            <p style="word-break: break-all; color: #666;">{verification_url}</p>
                            
                            <p>This verification link will expire in 1 hour.</p>
                            
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
            
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #007AFF;">PosturEase</h1>
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

# Create global email service instance
email_service = EmailService()
