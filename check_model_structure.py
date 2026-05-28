#!/usr/bin/env python3
"""
Check the actual model structure to understand feature mismatch
"""

import joblib
import numpy as np

def check_model_structure():
    """Check the actual model structure and expected features"""
    try:
        # Load both original and corrected assets
        original_assets = joblib.load("models/loan_prediction_assets_1.pkl")
        corrected_assets = joblib.load("models/loan_prediction_assets_1_fixed.pkl")
        
        print("=== Original Assets ===")
        print(f"Features: {len(original_assets['features'])}")
        print(original_assets['features'])
        
        print("\n=== Corrected Assets ===")
        print(f"Features: {len(corrected_assets['features'])}")
        print(corrected_assets['features'])
        
        # Check model itself
        original_model = original_assets['model']
        corrected_model = corrected_assets['model']
        
        print(f"\n=== Model Info ===")
        print(f"Original model feature count: {original_model.n_features_in_}")
        print(f"Corrected model feature count: {corrected_model.n_features_in_}")
        
        # The issue: model still expects 25 features but we only provide 24
        if original_model.n_features_in_ != len(corrected_assets['features']):
            print(f"\nPROBLEM: Model expects {original_model.n_features_in_} features but assets only have {len(corrected_assets['features'])}")
            
            # Find the missing feature
            original_features = set(original_assets['features'])
            corrected_features = set(corrected_assets['features'])
            missing_features = original_features - corrected_features
            extra_features = corrected_features - original_features
            
            print(f"Missing features: {missing_features}")
            print(f"Extra features: {extra_features}")
            
            return missing_features
        
        return None
        
    except Exception as e:
        print(f"Error checking model structure: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    missing = check_model_structure()
    
    if missing:
        print(f"\nNeed to handle missing feature: {missing}")
        print("Options:")
        print("1. Add the missing feature back to the data processing")
        print("2. Retrain the model without the feature")
        print("3. Use a different model file")

if __name__ == "__main__":
    main()
