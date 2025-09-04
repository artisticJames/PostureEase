from db import get_db_connection
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_email_verification():
    """Add email verification columns to users table"""
    try:
        conn = get_db_connection()
        if not conn:
            print("Failed to connect to database")
            return False

        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = DATABASE() 
            AND TABLE_NAME = 'users' 
            AND COLUMN_NAME IN ('email_verified', 'verification_token', 'token_expires', 'password_change_token', 'password_token_expires')
        """)
        
        existing_columns = [row[0] for row in cursor.fetchall()]
        
        # Add email verification columns if they don't exist
        if 'email_verified' not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT FALSE")
            print("Added email_verified column")
        
        if 'verification_token' not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN verification_token VARCHAR(255) NULL")
            print("Added verification_token column")
        
        if 'token_expires' not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN token_expires DATETIME NULL")
            print("Added token_expires column")
        
        if 'password_change_token' not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN password_change_token VARCHAR(255) NULL")
            print("Added password_change_token column")
        
        if 'password_token_expires' not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN password_token_expires DATETIME NULL")
            print("Added password_token_expires column")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("Database migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        print(f"Migration failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting email verification database migration...")
    migrate_email_verification()
