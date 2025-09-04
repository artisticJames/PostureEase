import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Database configuration
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'james-2003')
    MYSQL_DB = os.getenv('MYSQL_DB', 'posturease')
    
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24))
    
    # Email configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '465'))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', 'your-email@gmail.com')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', 'your-app-password')
    FROM_EMAIL = os.getenv('FROM_EMAIL', 'your-email@gmail.com')
    BASE_URL = os.getenv('BASE_URL', 'http://localhost:5000')
    
    # Other configurations can be added here 