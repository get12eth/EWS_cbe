#!/usr/bin/env python3
"""
Create Prediction Results Table in Database
"""

import mysql.connector
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def create_prediction_results_table():
    """Create prediction_results table"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Create table SQL
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id INT,
            contract_code VARCHAR(50),
            
            -- Prediction results
            predicted_status VARCHAR(20), -- PAS, SME, SET, NPL
            npl_probability DECIMAL(8,6),
            pas_probability DECIMAL(8,6),
            sme_probability DECIMAL(8,6),
            set_probability DECIMAL(8,6),
            risk_level VARCHAR(20), -- Low Risk, Medium Risk, High Risk
            
            -- Metadata
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version VARCHAR(50) DEFAULT 'v1.0',
            feature_count INT DEFAULT 31,
            
            -- Audit fields
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Foreign key relationships
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (contract_code) REFERENCES customers(CONTRACT_CODE),
            
            -- Indexes for performance
            INDEX idx_contract_code (contract_code),
            INDEX idx_prediction_date (prediction_date),
            INDEX idx_predicted_status (predicted_status),
            INDEX idx_customer_id (customer_id)
        )
        """
        
        logger.info("Creating prediction_results table...")
        cursor.execute(create_table_sql)
        logger.info("✅ prediction_results table created successfully!")
        
        # Verify table creation
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = DATABASE() 
            AND TABLE_NAME = 'prediction_results'
        """)
        table_exists = cursor.fetchone()[0] > 0
        
        if table_exists:
            logger.info("✅ Table verification successful!")
            
            # Show table structure
            cursor.execute("DESCRIBE prediction_results")
            columns = cursor.fetchall()
            logger.info("Table structure:")
            for col in columns:
                logger.info(f"  - {col[0]}: {col[1]}")
        else:
            logger.error("❌ Table creation failed!")
            return False
        
        conn.commit()
        return True
        
    except Exception as e:
        logger.error(f"Error creating table: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def main():
    """Main function"""
    print("=" * 80)
    print("CREATE PREDICTION RESULTS TABLE")
    print("=" * 80)
    
    success = create_prediction_results_table()
    
    if success:
        print("\n✅ SUCCESS: prediction_results table created successfully!")
        print("   - Separate table for prediction results")
        print("   - Foreign key relationships to customers table")
        print("   - Indexes for optimal performance")
        print("\nNext steps:")
        print("   1. Update predict_customer_risk to store results in prediction_results")
        print("   2. Update get_customers to join with prediction_results")
        print("   3. Test prediction display in customer management")
    else:
        print("\n❌ FAILED: Error creating prediction_results table")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
