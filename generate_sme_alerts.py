#!/usr/bin/env python3
"""
Generate SME alerts with lower threshold for demonstration
"""

import mysql.connector
from datetime import datetime

def get_db_connection():
    """Get database connection"""
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='Bant@6963',
        database='lon-default'
    )

def generate_sme_alerts():
    """Generate SME alerts with lower threshold"""
    print("Generating SME Alerts with Lower Threshold")
    print("=" * 50)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get customers with any SME probability (lower threshold)
        cursor.execute("""
            SELECT CONTRACT_CODE, SME_PROBABILITY, id, PAS_PROBABILITY, SET_PROBABILITY
            FROM customers 
            WHERE SME_PROBABILITY > 0.1
            ORDER BY SME_PROBABILITY DESC
        """)
        
        sme_customers = cursor.fetchall()
        print(f"Found {len(sme_customers)} customers with SME probability > 0.1")
        
        alerts_created = 0
        
        for contract_code, sme_prob, customer_id, pas_prob, set_prob in sme_customers:
            try:
                # Determine severity based on probability
                if sme_prob > 0.25:
                    severity = 'high'
                elif sme_prob > 0.20:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                # Create alert
                alert_sql = """
                    INSERT INTO alerts 
                    (entity_id, risk_signal, severity, prediction_score, status, 
                     contract_code, customer_name, alert_timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(alert_sql, (
                    contract_code,
                    'SME Prediction Alert',
                    severity,
                    sme_prob,
                    'open',
                    contract_code,
                    f'Customer {contract_code}',
                    datetime.now()
                ))
                
                alerts_created += 1
                print(f"Created {severity} SME alert for {contract_code} (SME: {sme_prob:.4f}, PAS: {pas_prob:.4f}, SET: {set_prob:.4f})")
                
            except Exception as e:
                print(f"Error creating alert for {contract_code}: {e}")
                continue
        
        conn.commit()
        print(f"\nCreated {alerts_created} SME alerts")
        
        # Check total alerts now
        cursor.execute("SELECT COUNT(*) FROM alerts")
        total_alerts = cursor.fetchone()[0]
        print(f"Total alerts in database: {total_alerts}")
        
        # Show sample alerts
        cursor.execute("""
            SELECT entity_id, risk_signal, severity, prediction_score, status, alert_timestamp
            FROM alerts 
            WHERE risk_signal LIKE '%SME%'
            ORDER BY alert_timestamp DESC
            LIMIT 5
        """)
        
        sample_alerts = cursor.fetchall()
        print("\nSample SME alerts:")
        for alert in sample_alerts:
            print(f"  {alert[0]}: {alert[1]} ({alert[2]}) - {alert[4]} - {alert[5]}")
        
        cursor.close()
        conn.close()
        
        return alerts_created
        
    except Exception as e:
        print(f"Error generating SME alerts: {e}")
        return 0

def main():
    """Main function"""
    alerts_created = generate_sme_alerts()
    
    print("\n" + "=" * 50)
    print("FINAL SUMMARY:")
    print(f"SME alerts created: {alerts_created}")
    
    if alerts_created > 0:
        print("SUCCESS: SME Alerts Management should now be populated!")
        print("Refresh the Alerts Management page to see the new alerts.")
        print("The alerts show customers with varying SME probability levels.")
    else:
        print("No SME alerts were created")

if __name__ == "__main__":
    main()
