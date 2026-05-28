"""
Create an admin user in the `lon-default` database.

Usage (from project root, after installing requirements):
python scripts/create_admin.py admin_username
It will prompt for a password and insert a hashed password into the users table.

Make sure MySQL is running and that `sql/create_users_table.sql` has been executed in MySQL Workbench.
"""
import getpass
import sys
import mysql.connector
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DB_CONF = dict(host='localhost', user='root', password='Bant@6963', database='lon-default')


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_admin.py <username>")
        return
    username = sys.argv[1]
    # Allow supplying password as second arg for non-interactive use
    if len(sys.argv) >= 3:
        password = sys.argv[2]
    else:
        password = getpass.getpass("Password for %s: " % username)
    pw_hash = pwd_context.hash(password)

    conn = mysql.connector.connect(**DB_CONF)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)", (username, pw_hash, 'admin'))
        conn.commit()
        print("Inserted user:", username)
    except mysql.connector.Error as e:
        print("Error inserting user:", e)
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    main()
