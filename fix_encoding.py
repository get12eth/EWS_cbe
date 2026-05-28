# Fix encoding issue in MySQL connection
import mysql.connector

def test_connection():
    try:
        # Test connection with proper encoding
        conn = mysql.connector.connect(
            host='localhost',
            user='root', 
            password='Bant@6963',
            database='lon-default',
            charset='utf8mb4',
            collation='utf8mb4_unicode_ci'
        )
        print("✅ Connection successful!")
        
        # Test the cleanup commands
        cursor = conn.cursor()
        
        # Check if columns exist
        cursor.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'lon-default' 
            AND TABLE_NAME = 'customers' 
            AND COLUMN_NAME IN ('LOAN_STATUS', 'NBE_LOAN_STATUS')
        """)
        columns = cursor.fetchall()
        print(f"Found columns: {columns}")
        
        # Drop columns if they exist
        for column in columns:
            column_name = column[0]
            try:
                cursor.execute(f"ALTER TABLE customers DROP COLUMN {column_name}")
                print(f"✅ Dropped column: {column_name}")
            except Exception as e:
                print(f"❌ Error dropping {column_name}: {e}")
        
        # Drop index if exists
        try:
            cursor.execute("DROP INDEX idx_customers_status ON customers")
            print("✅ Dropped index: idx_customers_status")
        except Exception as e:
            print(f"❌ Error dropping index: {e}")
            
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Cleanup completed successfully!")
        
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    test_connection()
