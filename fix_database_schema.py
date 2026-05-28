import mysql.connector
from mysql.connector import Error

def fix_database_schema():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root', 
            password='Bant@6963', 
            database='lon-default'
        )
        cursor = conn.cursor()
        
        # Create missing model_performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(100),
                model_version VARCHAR(50),
                evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accuracy_score DECIMAL(5,4),
                precision_score DECIMAL(5,4),
                recall_score DECIMAL(5,4),
                f1_score DECIMAL(5,4),
                auc_roc DECIMAL(5,4),
                confusion_matrix JSON,
                feature_importance JSON,
                training_samples INT,
                test_samples INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add missing columns to data_validation_logs
        cursor.execute('SHOW COLUMNS FROM data_validation_logs LIKE "alert_id"')
        result = cursor.fetchone()
        if not result:
            cursor.execute('ALTER TABLE data_validation_logs ADD COLUMN alert_id VARCHAR(50)')
        
        # Add missing columns to alerts
        cursor.execute('SHOW COLUMNS FROM alerts LIKE "contract_code"')
        result = cursor.fetchone()
        if not result:
            cursor.execute('ALTER TABLE alerts ADD COLUMN contract_code VARCHAR(50)')
            
        cursor.execute('SHOW COLUMNS FROM alerts LIKE "customer_name"')
        result = cursor.fetchone()
        if not result:
            cursor.execute('ALTER TABLE alerts ADD COLUMN customer_name VARCHAR(100)')
        
        conn.commit()
        print('Database schema updated successfully')
        
    except Error as e:
        print(f'Database error: {e}')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    fix_database_schema()
