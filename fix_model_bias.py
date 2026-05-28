#!/usr/bin/env python3
"""
Fix Model Bias - Investigate and Correct Prediction Bias
"""

import pandas as pd
import numpy as np
import mysql.connector
import joblib
import logging
from collections import Counter
import matplotlib.pyplot as plt

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

def analyze_training_data_distribution():
    """Analyze the actual distribution of training data in the database"""
    conn = get_db_connection()
    try:
        # Get all customers with their predicted status
        query = """
        SELECT PREDICTED_STATUS, COUNT(*) as count
        FROM customers 
        WHERE PREDICTED_STATUS IS NOT NULL
        GROUP BY PREDICTED_STATUS
        ORDER BY count DESC
        """
        df = pd.read_sql(query, conn)
        
        print("Training Data Distribution:")
        print(df.to_string(index=False))
        
        # Get detailed statistics
        detailed_query = """
        SELECT 
            PREDICTED_STATUS,
            COUNT(*) as count,
            AVG(APPROVED_AMOUNT) as avg_amount,
            AVG(PRINCIPAL_OS) as avg_principal,
            AVG(NPL_PROBABILITY) as avg_npl_prob,
            MIN(NPL_PROBABILITY) as min_npl_prob,
            MAX(NPL_PROBABILITY) as max_npl_prob
        FROM customers 
        WHERE PREDICTED_STATUS IS NOT NULL
        GROUP BY PREDICTED_STATUS
        ORDER BY count DESC
        """
        detailed_df = pd.read_sql(detailed_query, conn)
        
        print("\nDetailed Statistics by Class:")
        print(detailed_df.to_string(index=False))
        
        return df, detailed_df
    except Exception as e:
        logger.error(f"Error analyzing training data: {e}")
        return None, None
    finally:
        conn.close()

def create_balanced_test_data():
    """Create balanced test data to check model behavior"""
    conn = get_db_connection()
    try:
        # Get sample customers from each class if available
        query = """
        SELECT * FROM customers 
        WHERE PREDICTED_STATUS IS NOT NULL
        AND APPROVED_AMOUNT IS NOT NULL AND APPROVED_AMOUNT > 0
        ORDER BY PREDICTED_STATUS, CREATED_AT DESC
        """
        df = pd.read_sql(query, conn)
        
        print(f"Total customers found: {len(df)}")
        print(f"Class distribution in full dataset:")
        print(df['PREDICTED_STATUS'].value_counts())
        
        return df
    except Exception as e:
        logger.error(f"Error creating test data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def simulate_customer_scenarios():
    """Create synthetic customer scenarios to test model diversity"""
    scenarios = {
        'Low Risk Customer': {
            'APPROVED_AMOUNT': 50000,
            'PRINCIPAL_OS': 25000,
            'PRINCIPAL_ARREARS': 0,
            'INTEREST_ARREARS': 0,
            'RISK_GRADE': 'A',
            'ECONOMIC_SECTOR': 'Agriculture',
            'LOAN_TYPE': 'Personal',
            'TENURE': '12 months'
        },
        'Medium Risk Customer': {
            'APPROVED_AMOUNT': 100000,
            'PRINCIPAL_OS': 80000,
            'PRINCIPAL_ARREARS': 5000,
            'INTEREST_ARREARS': 1000,
            'RISK_GRADE': 'B',
            'ECONOMIC_SECTOR': 'Manufacturing',
            'LOAN_TYPE': 'Business',
            'TENURE': '24 months'
        },
        'High Risk Customer': {
            'APPROVED_AMOUNT': 200000,
            'PRINCIPAL_OS': 180000,
            'PRINCIPAL_ARREARS': 20000,
            'INTEREST_ARREARS': 5000,
            'RISK_GRADE': 'C',
            'ECONOMIC_SECTOR': 'Construction',
            'LOAN_TYPE': 'Commercial',
            'TENURE': '36 months'
        },
        'Very High Risk Customer': {
            'APPROVED_AMOUNT': 500000,
            'PRINCIPAL_OS': 450000,
            'PRINCIPAL_ARREARS': 50000,
            'INTEREST_ARREARS': 15000,
            'RISK_GRADE': 'D',
            'ECONOMIC_SECTOR': 'Real Estate',
            'LOAN_TYPE': 'Mortgage',
            'TENURE': '60 months'
        }
    }
    
    return scenarios

def test_model_with_scenarios(scenarios):
    """Test model with different customer scenarios"""
    # Load model assets
    try:
        assets = joblib.load("models/loan_prediction_assets_.pkl")
        model = assets['model']
        label_encoder = assets['label_encoder']
        feature_cols = assets['features']
        te = assets['target_encoder']
        ord_enc = assets.get('ordinal_encoder')
        remaining_cats = assets.get('remaining_cats', [])
        
        print(f"\nTesting model with {len(scenarios)} scenarios...")
        print(f"Model classes: {label_encoder.classes_}")
        
        results = {}
        
        for scenario_name, scenario_data in scenarios.items():
            try:
                # Create a customer record with scenario data
                customer_record = {}
                
                # Fill with all required features (default values)
                for col in feature_cols:
                    customer_record[col] = 0
                
                # Override with scenario-specific values
                for key, value in scenario_data.items():
                    if key in customer_record:
                        customer_record[key] = value
                
                # Create DataFrame
                customer_df = pd.DataFrame([customer_record])
                
                # Prepare model input (simplified version)
                X = customer_df[feature_cols].copy()
                X = X.fillna(0)
                X_array = X.values.astype(np.float64)
                
                # Make prediction
                pred = model.predict(X_array)[0]
                pred_proba = model.predict_proba(X_array)[0]
                pred_label = label_encoder.classes_[pred]
                
                results[scenario_name] = {
                    'prediction': pred_label,
                    'probabilities': dict(zip(label_encoder.classes_, pred_proba)),
                    'highest_prob': np.max(pred_proba),
                    'scenario_data': scenario_data
                }
                
                print(f"\n{scenario_name}:")
                print(f"  Predicted: {pred_label}")
                print(f"  Probabilities: {dict(zip(label_encoder.classes_, [f'{p:.4f}' for p in pred_proba]))}")
                
            except Exception as e:
                logger.error(f"Error testing scenario {scenario_name}: {e}")
                results[scenario_name] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return {}

def investigate_model_training():
    """Investigate how the model was trained"""
    try:
        # Load model assets
        assets = joblib.load("models/loan_prediction_assets_.pkl")
        model = assets['model']
        label_encoder = assets['label_encoder']
        
        print(f"\nModel Investigation:")
        print(f"Model type: {type(model)}")
        print(f"Model parameters: {model.get_params()}")
        print(f"Label encoder classes: {label_encoder.classes_}")
        
        # Check if model has feature importance
        if hasattr(model, 'feature_importances_'):
            feature_cols = assets['features']
            importance_dict = dict(zip(feature_cols, model.feature_importances_))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 10 Feature Importances:")
            for feature, importance in sorted_importance[:10]:
                print(f"  {feature}: {importance:.4f}")
        
        # Check model's training parameters
        if hasattr(model, 'n_classes_'):
            print(f"Number of classes: {model.n_classes_}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error investigating model: {e}")
        return False

def main():
    """Main analysis function"""
    print("="*80)
    print("MODEL BIAS INVESTIGATION AND FIX")
    print("="*80)
    
    # 1. Analyze training data distribution
    print("\n1. TRAINING DATA DISTRIBUTION ANALYSIS")
    print("-" * 50)
    df, detailed_df = analyze_training_data_distribution()
    
    # 2. Investigate model training
    print("\n2. MODEL TRAINING INVESTIGATION")
    print("-" * 50)
    investigate_model_training()
    
    # 3. Test with synthetic scenarios
    print("\n3. SYNTHETIC SCENARIO TESTING")
    print("-" * 50)
    scenarios = simulate_customer_scenarios()
    scenario_results = test_model_with_scenarios(scenarios)
    
    # 4. Analyze results and provide recommendations
    print("\n4. ANALYSIS AND RECOMMENDATIONS")
    print("-" * 50)
    
    if df is not None:
        class_counts = df['count'].values
        total_count = np.sum(class_counts)
        
        if len(class_counts) < 4:
            print("❌ CRITICAL ISSUE: Training data doesn't contain all 4 classes!")
            print("   The model cannot predict classes it wasn't trained on.")
            print("\nRecommendations:")
            print("   1. Collect more diverse training data")
            print("   2. Balance the training dataset")
            print("   3. Consider data augmentation for underrepresented classes")
        else:
            # Check for class imbalance
            max_count = np.max(class_counts)
            min_count = np.min(class_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 3:
                print(f"⚠️  CLASS IMBALANCE DETECTED (ratio: {imbalance_ratio:.1f}:1)")
                print("   Model may be biased toward majority classes")
                print("\nRecommendations:")
                print("   1. Use class weights in training")
                print("   2. Oversample minority classes")
                print("   3. Use balanced accuracy metrics")
            else:
                print("✅ Training data appears reasonably balanced")
    
    # Check scenario results
    diverse_predictions = set()
    for scenario_name, result in scenario_results.items():
        if 'prediction' in result:
            diverse_predictions.add(result['prediction'])
    
    if len(diverse_predictions) == 1:
        print("❌ MODEL BIAS CONFIRMED: Model only predicts one class")
        print("   Even with diverse input scenarios, output is uniform")
        print("\nImmediate Actions Required:")
        print("   1. Retrain model with balanced dataset")
        print("   2. Check for data leakage in training process")
        print("   3. Validate model training pipeline")
    elif len(diverse_predictions) < 4:
        print(f"⚠️  PARTIAL BIAS: Model only predicts {len(diverse_predictions)}/4 classes")
        print(f"   Missing classes: {set(['NPL', 'PAS', 'SET', 'SME']) - diverse_predictions}")
    else:
        print("✅ Model shows diverse predictions in scenarios")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
