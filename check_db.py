from config import Config
import mysql.connector as mc
print('CFG_DB=', Config.MYSQL_DB)
print('CFG_HOST=', Config.MYSQL_HOST)
conn = mc.connect(host=Config.MYSQL_HOST, user=Config.MYSQL_USER, password=Config.MYSQL_PASSWORD, database=Config.MYSQL_DB, auth_plugin='mysql_native_password')
cur = conn.cursor()
cur.execute("SHOW TABLES LIKE 'otp_verifications'")
print('OTP_TABLE_EXISTS=', bool(cur.fetchone()))
cur.execute('SELECT DATABASE()')
print('CONNECTED_DB=', cur.fetchone()[0])
cur.close(); conn.close()
