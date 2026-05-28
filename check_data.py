#!/usr/bin/env python3
"""
Check customer and prediction data
"""

import mysql.connector

def get_db_connection():
    """Connect to database"""
    return mysql.connector.connect(
        host='localhost',
        user='root', 
        password='Bant@6963',
        database='lon-default',
        charset='utf8mb4',
        collation='utf8mb4_unicode_ci'
    )

def check_data():
    """Check customer and prediction data"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Check customers
        cursor.execute('SELECT id, CONTRACT_CODE, CUST_SHORTNAME FROM customers ORDER BY id LIMIT 3')
        customers = cursor.fetchall()
        print('Sample customers:')
        for c in customers:
            print(f'  ID: {c["id"]}, Contract: {c["CONTRACT_CODE"]}, Name: {c["CUST_SHORTNAME"]}')
        
        # Check predictions
        cursor.execute('SELECT customer_id, predicted_status, risk_level, prediction_date FROM prediction_results ORDER BY prediction_date DESC LIMIT 3')
        predictions = cursor.fetchall()
        print('\nRecent predictions:')
        for p in predictions:
            print(f'  Customer ID: {p["customer_id"]}, Status: {p["predicted_status"]}, Risk: {p["risk_level"]}, Date: {p["prediction_date"]}')
        
        # Test prediction for first customer
        if customers:
            customer_id = customers[0]['id']
            print(f'\nTesting prediction for customer {customer_id}...')
            
            # Import main to test prediction
            import sys
            sys.path.append('.')
            from main import predict_customer_risk
            
            try:
                result = predict_customer_risk(customer_id)
                print(f'Prediction result: {result}')
            except Exception as e:
                print(f'Prediction error: {e}')
        
    except Exception as e:
        print(f'Error: {e}')
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    check_data()
