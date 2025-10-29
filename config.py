import os
from dotenv import load_dotenv

# Ensure we load the .env located next to this file, regardless of cwd
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_BASE_DIR, '.env')
load_dotenv(dotenv_path=_ENV_PATH)

# Fallback: also load from CWD if present (harmless if already loaded)
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
    
    
    # Fixed posture classification threshold (no user-configurable sensitivity)
    POSTURE_GOOD_CONFIDENCE_THRESHOLD = 0.7
    
    # Session summary thresholds (percent in [0,1])
    SESSION_GOOD_THRESHOLD = float(os.getenv('SESSION_GOOD_THRESHOLD', '0.8'))
    SESSION_FAIR_THRESHOLD = float(os.getenv('SESSION_FAIR_THRESHOLD', '0.6'))
    
    # Camera configuration
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '-1'))  # -1 for auto-detection, 0,1,2,etc for specific camera
    CAMERA_MIRROR = os.getenv('CAMERA_MIRROR', 'true').lower() == 'true'  # Enable/disable camera mirroring
    
    # Other configurations can be added here 
    
    # Gemini API configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyD6jHKKEurfH81xKTp-LUdk_I7EuUFRt7I')

    # EmailJS configuration
    EMAILJS_SERVICE_ID = os.getenv('EMAILJS_SERVICE_ID', 'service_t29owva')
    EMAILJS_PUBLIC_KEY = os.getenv('EMAILJS_PUBLIC_KEY', 'Ue9cupo5ozzHLwZs-')
    EMAILJS_PRIVATE_KEY = os.getenv('EMAILJS_PRIVATE_KEY', 'ZY5_sxR8iGPSw9HHEDGuA')
    EMAILJS_TEMPLATE_ID_VERIFY = os.getenv('EMAILJS_TEMPLATE_ID_VERIFY', 'template_t8e5ukn')
    EMAILJS_TEMPLATE_ID_RESET = os.getenv('EMAILJS_TEMPLATE_ID_RESET', 'template_t8e5ukn')