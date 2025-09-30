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
    
    # ML Model configuration
    USE_ML_CLASSIFIER = os.getenv('USE_ML_CLASSIFIER', 'true').lower() == 'true'
    ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', 'Training(Updated)')
    
    # Posture classification thresholds
    # Per-frame confidence threshold to label as Good vs Bad
    # Manager requirement: 70% threshold (0.7) - below 70% = bad posture, above 70% = good posture
    POSTURE_GOOD_CONFIDENCE_THRESHOLD = float(os.getenv('POSTURE_GOOD_CONFIDENCE_THRESHOLD', '0.7'))
    
    # Available confidence levels for user selection
    CONFIDENCE_LEVELS = {
        '50%': 0.5,
        '70%': 0.7,
        '80%': 0.8,
        '90%': 0.9,
        '95%': 0.95
    }
    
    # Session summary thresholds (percent in [0,1])
    SESSION_GOOD_THRESHOLD = float(os.getenv('SESSION_GOOD_THRESHOLD', '0.8'))
    SESSION_FAIR_THRESHOLD = float(os.getenv('SESSION_FAIR_THRESHOLD', '0.6'))
    
    # Camera configuration
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '-1'))  # -1 for auto-detection, 0,1,2,etc for specific camera
    CAMERA_MIRROR = os.getenv('CAMERA_MIRROR', 'true').lower() == 'true'  # Enable/disable camera mirroring
    
    # Other configurations can be added here 