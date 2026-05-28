#!/usr/bin/env python3
"""
Script to create admin user with hashed password
"""
import mysql.connector
from passlib.context import CryptContext

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host='localhost', 
        user='root', 
        password='Bant@6963', 
        database='lon-default'
    )

# Password hashing
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def create_admin_user():
    username = "admin"
    password = "admin123"
    
    # Hash the password
    password_hash = pwd_context.hash(password)
    
    print(f"Creating admin user...")
    print(f"Username: {username}")
    print(f"Password: {password}")
    print(f"Password hash: {password_hash}")
    
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if admin user already exists
        cur.execute('SELECT id FROM users WHERE username = %s', (username,))
        existing_user = cur.fetchone()
        
        if existing_user:
            # Update existing user
            cur.execute('UPDATE users SET password_hash = %s WHERE username = %s', 
                       (password_hash, username))
            print("Updated existing admin user password")
        else:
            # Insert new user
            cur.execute('INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)',
                       (username, password_hash, 'admin'))
            print("Created new admin user")
        
        conn.commit()
        print("Admin user created successfully!")
        
        # Verify the user
        cur.execute('SELECT username, password_hash, role FROM users WHERE username = %s', (username,))
        user = cur.fetchone()
        if user:
            print(f"Verification - User found: {user[0]}, Role: {user[2]}")
            
            # Test password verification
            if pwd_context.verify(password, user[1]):
                print("Password verification: SUCCESS")
            else:
                print("Password verification: FAILED")
        
    except Exception as e:
        print(f"Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_admin_user()
