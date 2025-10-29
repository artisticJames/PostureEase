#!/usr/bin/env python3
"""
Database migration script to add OTP verification table
Run this script to add the otp_verifications table to your database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import get_db_connection
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_otp_table():
    """Add OTP verification table to the database"""
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("Failed to connect to database")
            return False

        cursor = conn.cursor()
        
        # Create OTP verifications table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS otp_verifications (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) NOT NULL,
                otp_code VARCHAR(6) NOT NULL,
                verification_type ENUM('registration', 'password_change', 'email_change') NOT NULL DEFAULT 'registration',
                expires_at DATETIME NOT NULL,
                created_at DATETIME NOT NULL,
                INDEX idx_email_type (email, verification_type),
                INDEX idx_expires (expires_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("OTP verifications table created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error creating OTP table: {e}")
        return False

if __name__ == "__main__":
    print("Adding OTP verification table...")
    success = migrate_otp_table()
    
    if success:
        print("Migration completed successfully!")
        print("OTP verification table has been added to your database.")
    else:
        print("Migration failed. Please check the error messages above.")
        sys.exit(1)
