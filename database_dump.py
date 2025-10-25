#!/usr/bin/env python3
"""
PostureEase Database Dump Script
Exports all database data to SQL files for team sharing
"""

from db import get_db_connection
import json
from datetime import datetime

def export_database():
    """Export all database tables to SQL files"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        print(f"Found {len(tables)} tables: {tables}")
        
        # Create SQL dump file
        with open('database_dump.sql', 'w', encoding='utf-8') as f:
            f.write("-- PostureEase Database Dump\n")
            f.write(f"-- Generated on: {datetime.now()}\n")
            f.write("-- This file contains all data from your PostureEase database\n\n")
            
            f.write("-- Create database\n")
            f.write("CREATE DATABASE IF NOT EXISTS posturease;\n")
            f.write("USE posturease;\n\n")
            
            for table in tables:
                print(f"Exporting table: {table}")
                
                # Get table structure
                cursor.execute(f"SHOW CREATE TABLE {table}")
                create_table = cursor.fetchone()[1]
                f.write(f"-- Table: {table}\n")
                f.write(f"{create_table};\n\n")
                
                # Get table data
                cursor.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()
                
                if rows:
                    # Get column names
                    cursor.execute(f"DESCRIBE {table}")
                    columns = [col[0] for col in cursor.fetchall()]
                    
                    f.write(f"-- Data for table: {table}\n")
                    f.write(f"INSERT INTO {table} ({', '.join(columns)}) VALUES\n")
                    
                    for i, row in enumerate(rows):
                        # Format values for SQL
                        formatted_values = []
                        for value in row:
                            if value is None:
                                formatted_values.append('NULL')
                            elif isinstance(value, str):
                                # Escape single quotes
                                escaped = value.replace("'", "''")
                                formatted_values.append(f"'{escaped}'")
                            elif isinstance(value, (int, float)):
                                formatted_values.append(str(value))
                            else:
                                formatted_values.append(f"'{str(value)}'")
                        
                        values_str = f"({', '.join(formatted_values)})"
                        if i < len(rows) - 1:
                            values_str += ","
                        else:
                            values_str += ";"
                        
                        f.write(f"{values_str}\n")
                    
                    f.write(f"\n-- {len(rows)} records inserted into {table}\n\n")
                else:
                    f.write(f"-- No data in {table}\n\n")
        
        print("Database export completed successfully!")
        print("File created: database_dump.sql")
        
        conn.close()
        
    except Exception as e:
        print(f"Error exporting database: {e}")

if __name__ == "__main__":
    export_database()
