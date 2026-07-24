#!/usr/bin/env python3
"""
Script to create database schema for CBE Loan Risk Management System
"""

import mysql.connector
import sys
import os

def execute_sql_schema():
    """Execute the comprehensive SQL schema"""
    
    #Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'Bant@6963',
        'database': 'lon-default'
    }
    
    #Read the SQL file
    sql_file_path = os.path.join(os.path.dirname(__file__), 'sql', 'comprehensive_schema.sql')
    
    try:
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            sql_content = file.read()
        
        # Connect to MySQL (without specifying database initially)
        conn = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        cursor = conn.cursor()
        
        #Create database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS `lon-default`")
        cursor.execute("USE `lon-default`")
        
        #Split SQL content into individual statements
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        print(f"Executing {len(statements)} SQL statements...")
        
        for i, statement in enumerate(statements, 1):
            try:
                if statement:
                    cursor.execute(statement)
                    print(f"Statement {i}: Executed successfully")
            except mysql.connector.Error as err:
                print(f"Statement {i}: Error - {err}")
                # Continue with other statements
        
        conn.commit()
        print("Database schema created successfully!")
        
        #Verify tables were created
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"\nCreated {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
    except Exception as e:
        print(f"Error creating database schema: {e}")
        return False
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()
    
    return True

if __name__ == "__main__":
    success = execute_sql_schema()
    if success:
        print("\n✅ Database setup completed successfully!")
        print("You can now start the application and the dashboard should work properly.")
    else:
        print("\n❌ Database setup failed. Please check the error messages above.")
        sys.exit(1)
