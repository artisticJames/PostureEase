import mysql.connector
from datetime import datetime
from config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLogger:
    def __init__(self):
        self.db_config = {
            'host': Config.MYSQL_HOST,
            'user': Config.MYSQL_USER,
            'password': Config.MYSQL_PASSWORD,
            'database': Config.MYSQL_DB
        }
        self.create_security_logs_table()
    
    def get_db_connection(self):
        """Get database connection"""
        return mysql.connector.connect(**self.db_config)
    
    def create_security_logs_table(self):
        """Create security logs table if it doesn't exist"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS security_logs (
                id INT PRIMARY KEY AUTO_INCREMENT,
                event_type ENUM('face_registered', 'face_recognized', 'unknown_user_detected', 'unauthorized_access', 'system_error') NOT NULL,
                user_id INT NULL,
                username VARCHAR(255) NULL,
                details TEXT NULL,
                ip_address VARCHAR(45) NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_event_type (event_type),
                INDEX idx_user_id (user_id),
                INDEX idx_timestamp (timestamp)
            );
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            logger.info("✅ Security logs table created/verified")
            
        except Exception as e:
            logger.error(f"❌ Error creating security logs table: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
    
    def log_event(self, event_type, user_id=None, username=None, details=None, ip_address=None):
        """Log a security event"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            query = """
            INSERT INTO security_logs (event_type, user_id, username, details, ip_address)
            VALUES (%s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (event_type, user_id, username, details, ip_address))
            conn.commit()
            
            logger.info(f"🔒 Security event logged: {event_type} - User: {username or 'Unknown'}")
            
        except Exception as e:
            logger.error(f"❌ Error logging security event: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
    
    def log_face_registered(self, user_id, username, ip_address=None):
        """Log face registration event"""
        self.log_event(
            event_type='face_registered',
            user_id=user_id,
            username=username,
            details=f"Face registered for user {username}",
            ip_address=ip_address
        )
    
    def log_face_recognized(self, user_id, username, confidence=None, ip_address=None):
        """Log face recognition event"""
        details = f"Face recognized for user {username}"
        if confidence:
            details += f" (confidence: {confidence:.2f})"
        
        self.log_event(
            event_type='face_recognized',
            user_id=user_id,
            username=username,
            details=details,
            ip_address=ip_address
        )
    
    def log_unknown_user_detected(self, ip_address=None):
        """Log unknown user detection event"""
        self.log_event(
            event_type='unknown_user_detected',
            details="Unknown user detected in camera feed",
            ip_address=ip_address
        )
    
    def log_unauthorized_access(self, username=None, details=None, ip_address=None):
        """Log unauthorized access attempt"""
        self.log_event(
            event_type='unauthorized_access',
            username=username,
            details=details or "Unauthorized access attempt",
            ip_address=ip_address
        )
    
    def log_system_error(self, error_details, ip_address=None):
        """Log system error"""
        self.log_event(
            event_type='system_error',
            details=error_details,
            ip_address=ip_address
        )
    
    def get_recent_security_events(self, limit=50):
        """Get recent security events"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            query = """
            SELECT * FROM security_logs 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            
            cursor.execute(query, (limit,))
            events = cursor.fetchall()
            
            return events
            
        except Exception as e:
            logger.error(f"❌ Error getting security events: {e}")
            return []
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
    
    def get_security_stats(self, days=7):
        """Get security statistics for the last N days"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            query = """
            SELECT 
                event_type,
                COUNT(*) as count,
                DATE(timestamp) as date
            FROM security_logs 
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY event_type, DATE(timestamp)
            ORDER BY date DESC, event_type
            """
            
            cursor.execute(query, (days,))
            stats = cursor.fetchall()
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting security stats: {e}")
            return []
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

# Global instance
security_logger = SecurityLogger()
