#!/usr/bin/env python3
"""
Clean up duplicate prediction results and test corrected prediction system
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Bant@6963',
    'database': 'lon-default'
}

def get_db_connection():
    """Create database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"Database connection error: {e}")
        return None

def cleanup_duplicate_predictions():
    """Remove duplicate prediction results"""
    print("=== Cleaning Up Duplicate Predictions ===")
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Check for duplicates before cleanup
        cursor.execute("SELECT contract_code, COUNT(*) as count FROM prediction_results GROUP BY contract_code HAVING count > 1")
        duplicates = cursor.fetchall()
        
        print(f"Found {len(duplicates)} contract codes with duplicates:")
        for contract_code, count in duplicates:
            print(f"  {contract_code}: {count} entries")
        
        # Remove duplicates, keeping the latest entry
        cursor.execute("""
            DELETE t1 FROM prediction_results t1
            INNER JOIN prediction_results t2 
            WHERE t1.id > t2.id 
            AND t1.contract_code = t2.contract_code
        """)
        
        conn.commit()
        
        # Verify cleanup
        cursor.execute("SELECT contract_code, COUNT(*) as count FROM prediction_results GROUP BY contract_code HAVING count > 1")
        remaining_duplicates = cursor.fetchall()
        
        if len(remaining_duplicates) == 0:
            print("All duplicates removed successfully!")
        else:
            print(f"Still have {len(remaining_duplicates)} duplicates remaining")
        
        return True
        
    except Error as e:
        print(f"Error cleaning up duplicates: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def test_corrected_prediction():
    """Test prediction with corrected model assets"""
    print("\n=== Testing Corrected Prediction System ===")
    
    try:
        # Import the corrected main module
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from main import prepare_model_input, assets
        
        print(f"Model expects {len(assets['features'])} features:")
        for i, feature in enumerate(assets['features']):
            print(f"  {i+1}. {feature}")
        
        # Get sample customer data
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers LIMIT 3")
        rows = cursor.fetchall()
        
        # Get column names
        cursor.execute("DESCRIBE customers")
        columns = [col[0] for col in cursor.fetchall()]
        
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        print(f"\nTesting with {len(df)} sample customers")
        
        # Test prepare_model_input
        X_array = prepare_model_input(df)
        
        print(f"Prepared input shape: {X_array.shape}")
        print(f"Input data type: {X_array.dtype}")
        
        # Check if rows are different now
        if X_array.shape[0] > 1:
            diff_count = np.sum(X_array[0] != X_array[1])
            print(f"Differences between first two rows: {diff_count}/{X_array.shape[1]} features")
            
            if diff_count > 0:
                print("SUCCESS: Rows have different feature values!")
            else:
                print("WARNING: Rows still have identical feature values")
        
        # Test actual prediction
        model = assets['model']
        label_encoder = assets['label_encoder']
        
        # Enable categorical support for XGBoost
        if hasattr(model, 'enable_categorical'):
            model.enable_categorical = True
        
        # Make predictions
        predictions = model.predict(X_array)
        predicted_statuses = label_encoder.inverse_transform(predictions)
        
        print(f"\nPredictions for sample customers:")
        for i, (contract_code, prediction) in enumerate(zip(df['CONTRACT_CODE'], predicted_statuses)):
            print(f"  {contract_code}: {prediction}")
        
        # Check if predictions are different
        unique_predictions = np.unique(predicted_statuses)
        print(f"Unique predictions: {unique_predictions}")
        
        if len(unique_predictions) > 1:
            print("SUCCESS: Different customers get different predictions!")
        else:
            print("WARNING: All customers still get the same prediction")
        
        return True
        
    except Exception as e:
        print(f"Error testing corrected prediction: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()

def main():
    """Main function"""
    print("Testing Corrected Prediction System")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Clean up duplicates
    if cleanup_duplicate_predictions():
        success_count += 1
    
    # Test 2: Test corrected prediction
    if test_corrected_prediction():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("Prediction system is now working correctly!")
    else:
        print("Some issues remain - check the errors above")

if __name__ == "__main__":
    main()
