import mysql.connector
from mysql.connector import Error
from config import Config
import bcrypt
import logging
import time
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection(max_retries=3, retry_delay=1):
    """
    Attempt to establish a database connection with retries
    """
    for attempt in range(max_retries):
        try:
            connection = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                auth_plugin='mysql_native_password',
                connect_timeout=10,
                use_pure=True
            )
            if connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                return connection
        except Error as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} failed to connect to MySQL: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("All connection attempts failed")
                return None
    return None

def test_connection():
    """
    Test the database connection and return True if successful
    """
    try:
        conn = get_db_connection()
        if conn and conn.is_connected():
            conn.close()
            return True
        return False
    except Exception as e:
        logger.error(f"Error testing database connection: {e}")
        return False

def create_user(username, email, password, first_name, last_name, birth_date, gender):
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor()
        
        # Check if username or email already exists
        cursor.execute("SELECT username, email FROM users WHERE username = %s OR email = %s", (username, email))
        existing_user = cursor.fetchone()
        
        if existing_user:
            if existing_user[0] == username:
                return False, "Username already exists"
            else:
                return False, "Email already registered"
        
        # Hash the password and store as UTF-8 string for portability
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Insert user with email verification fields
        cursor.execute("""
            INSERT INTO users (username, email, password, first_name, last_name, birth_date, gender, email_verified, verification_token, token_expires)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (username, email, hashed_password, first_name, last_name, birth_date, gender, False, None, None))
        
        conn.commit()
        user_id = cursor.lastrowid
        
        cursor.close()
        conn.close()
        
        return True, user_id
    except Error as e:
        logger.error(f"Error creating user: {e}")
        return False, str(e)

def verify_user(username, password):
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor(dictionary=True)
        
        # Get user
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if user:
            stored_hash = user.get('password')
            # Ensure we have bytes for bcrypt
            if isinstance(stored_hash, str):
                stored_hash_bytes = stored_hash.encode('utf-8')
            else:
                stored_hash_bytes = stored_hash
            if stored_hash_bytes and bcrypt.checkpw(password.encode('utf-8'), stored_hash_bytes):
                # Remove password from user data
                user.pop('password', None)
                cursor.close()
                conn.close()
                return True, user
        
        cursor.close()
        conn.close()
        
        return False, "Invalid username or password"
    except Error as e:
        logger.error(f"Error verifying user: {e}")
        return False, str(e)

def save_posture_record(user_id, posture_type, confidence_score, session_duration=None, corrections_count=None, good_time=None, bad_time=None):
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO posture_records (user_id, posture_type, confidence_score, session_duration, corrections_count, good_time, bad_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (user_id, posture_type, confidence_score, session_duration, corrections_count, good_time, bad_time))
        
        conn.commit()
        record_id = cursor.lastrowid
        
        cursor.close()
        conn.close()
        
        return True, record_id
    except Error as e:
        logger.error(f"Error saving posture record: {e}")
        return False, str(e)

def get_user_posture_history(user_id, limit=100):
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT * FROM posture_records 
            WHERE user_id = %s 
            ORDER BY recorded_at DESC 
            LIMIT %s
        """, (user_id, limit))
        
        records = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return True, records
    except Error as e:
        logger.error(f"Error getting posture history: {e}")
        return False, str(e)

def clear_user_posture_history(user_id):
    """Delete all posture records for a given user and return number deleted"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor()
        cursor.execute("DELETE FROM posture_records WHERE user_id = %s", (user_id,))
        deleted_count = cursor.rowcount
        conn.commit()

        cursor.close()
        conn.close()

        return True, deleted_count
    except Error as e:
        logger.error(f"Error clearing posture history: {e}")
        return False, str(e)

def delete_posture_record(record_id, user_id):
    """Delete a specific posture record for a given user"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor()
        cursor.execute("DELETE FROM posture_records WHERE id = %s AND user_id = %s", (record_id, user_id))
        deleted_count = cursor.rowcount
        conn.commit()

        cursor.close()
        conn.close()

        if deleted_count > 0:
            return True, "Record deleted successfully"
        else:
            return False, "Record not found or access denied"
    except Error as e:
        logger.error(f"Error deleting posture record: {e}")
        return False, str(e)

def get_exercises():
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM exercises")
        exercises = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return True, exercises
    except Error as e:
        logger.error(f"Error getting exercises: {e}")
        return False, str(e)

def record_completed_exercise(user_id, exercise_id):
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_exercises (user_id, exercise_id)
            VALUES (%s, %s)
        """, (user_id, exercise_id))
        
        conn.commit()
        record_id = cursor.lastrowid
        
        cursor.close()
        conn.close()
        
        return True, record_id
    except Error as e:
        logger.error(f"Error recording completed exercise: {e}")
        return False, str(e)

def update_user(user_id, username, email, first_name, last_name, birth_date, gender):
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor(dictionary=True)
        # Check if the new username or email is already taken by another user
        cursor.execute("SELECT id FROM users WHERE (username = %s OR email = %s) AND id != %s", (username, email, user_id))
        existing = cursor.fetchone()
        if existing:
            return False, "Username or email already taken by another user"

        # Update user info
        cursor.execute("""
            UPDATE users SET username=%s, email=%s, first_name=%s, last_name=%s, birth_date=%s, gender=%s, updated_at=NOW()
            WHERE id=%s
        """, (username, email, first_name, last_name, birth_date, gender, user_id))
        conn.commit()

        # Fetch updated user info
        cursor.execute("SELECT id, username, email, first_name, last_name, birth_date, gender FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        return True, user
    except Error as e:
        logger.error(f"Error updating user: {e}")
        return False, str(e)

def get_all_users():
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id, username, email, first_name, last_name, 
                   birth_date, gender, created_at, updated_at
            FROM users
            WHERE username != 'admin'
            ORDER BY created_at DESC
        """)
        
        users = cursor.fetchall()
        
        # Add default status for users (since status column doesn't exist yet)
        for user in users:
            user['status'] = True  # Default to active
        
        cursor.close()
        conn.close()
        
        return True, users
    except Error as e:
        logger.error(f"Error getting users: {e}")
        return False, str(e)

def delete_user(user_id):
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor()
        
        # First delete related records (including face embeddings, guarded if table missing)
        cursor.execute("DELETE FROM posture_records WHERE user_id = %s", (user_id,))
        cursor.execute("DELETE FROM user_exercises WHERE user_id = %s", (user_id,))
        try:
            cursor.execute("DELETE FROM user_face_embeddings WHERE user_id = %s", (user_id,))
        except Error as e:
            # If table does not exist, ignore; otherwise re-raise
            if getattr(e, 'errno', None) in (1146,):  # ER_NO_SUCH_TABLE
                logger.warning("user_face_embeddings table missing; skipping delete")
            else:
                raise
        
        # Then delete the user
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return True, "User deleted successfully"
    except Error as e:
        logger.error(f"Error deleting user: {e}")
        return False, str(e)

def toggle_user_status(user_id, status):
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor()
        
        # Check if status column exists
        cursor.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = DATABASE() 
            AND TABLE_NAME = 'users' 
            AND COLUMN_NAME = 'status'
        """)
        
        status_column_exists = cursor.fetchone()
        
        if status_column_exists:
            cursor.execute("""
                UPDATE users 
                SET status = %s, updated_at = NOW()
                WHERE id = %s
            """, (status, user_id))
        else:
            # Status column doesn't exist, just update the timestamp
            cursor.execute("""
                UPDATE users 
                SET updated_at = NOW()
                WHERE id = %s
            """, (user_id,))
        
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return True, "User status updated successfully"
    except Error as e:
        logger.error(f"Error updating user status: {e}")
        return False, str(e)

def admin_create_user(username, email, password, first_name, last_name, birth_date, gender):
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor()
        
        # Check if username or email already exists
        cursor.execute("SELECT username, email FROM users WHERE username = %s OR email = %s", (username, email))
        existing_user = cursor.fetchone()
        
        if existing_user:
            if existing_user[0] == username:
                return False, "Username already exists"
            else:
                return False, "Email already registered"
        
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Insert user (without status column since it doesn't exist yet)
        cursor.execute("""
            INSERT INTO users (username, email, password, first_name, last_name, birth_date, gender)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (username, email, hashed_password, first_name, last_name, birth_date, gender))
        
        conn.commit()
        user_id = cursor.lastrowid
        
        cursor.close()
        conn.close()
        
        return True, user_id
    except Error as e:
        logger.error(f"Error creating user: {e}")
        return False, str(e) 

def save_verification_token(user_id, token, expires_at, token_type="registration"):
    """Save verification token for email verification or password change"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor()
        
        if token_type == "registration":
            cursor.execute("""
                UPDATE users SET verification_token = %s, token_expires = %s 
                WHERE id = %s
            """, (token, expires_at, user_id))
        else:  # password change
            cursor.execute("""
                UPDATE users SET password_change_token = %s, password_token_expires = %s 
                WHERE id = %s
            """, (token, expires_at, user_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True, "Token saved successfully"
    except Error as e:
        logger.error(f"Error saving verification token: {e}")
        return False, str(e)

def verify_email_token(token):
    """Verify email verification token"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id, username, email, token_expires 
            FROM users 
            WHERE verification_token = %s
        """, (token,))
        
        user = cursor.fetchone()
        
        if not user:
            return False, "Invalid verification token"
        
        # Check if token is expired
        if user['token_expires'] and user['token_expires'] < datetime.now():
            return False, "Verification token has expired"
        
        # Mark email as verified
        cursor.execute("""
            UPDATE users 
            SET email_verified = TRUE, verification_token = NULL, token_expires = NULL 
            WHERE id = %s
        """, (user['id'],))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True, user
    except Error as e:
        logger.error(f"Error verifying email token: {e}")
        return False, str(e)

def verify_password_change_token(token):
    """Verify password change token"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id, username, email, password_change_token, password_token_expires 
            FROM users 
            WHERE password_change_token = %s
        """, (token,))
        
        user = cursor.fetchone()
        
        if not user:
            return False, "Invalid password change token"
        
        # Check if token is expired
        if user['password_token_expires'] and user['password_token_expires'] < datetime.now():
            return False, "Password change token has expired"
        
        return True, user
    except Error as e:
        logger.error(f"Error verifying password change token: {e}")
        return False, str(e)

def clear_password_change_token(user_id):
    """Clear password change token after successful password change"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET password_change_token = NULL, password_token_expires = NULL 
            WHERE id = %s
        """, (user_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True, "Token cleared successfully"
    except Error as e:
        logger.error(f"Error clearing password change token: {e}")
        return False, str(e)

def get_user_by_email(email):
    """Get user by email address"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if user:
            # Remove password from user data
            user.pop('password', None)
            return True, user
        
        return False, "User not found"
    except Error as e:
        logger.error(f"Error getting user by email: {e}")
        return False, str(e) 