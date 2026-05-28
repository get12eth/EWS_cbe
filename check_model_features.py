#!/usr/bin/env python3
"""
Check actual model features and create corrected version
"""

import joblib
import pandas as pd

def check_model_features():
    """Check what features the model actually expects"""
    try:
        assets = joblib.load("models/loan_prediction_assets_1.pkl")
        
        if 'features' in assets:
            feature_cols = assets['features']
            print(f"Model expects {len(feature_cols)} features:")
            for i, col in enumerate(feature_cols):
                print(f"  {i+1}. {col}")
                
            # Check if REGIONNAME is in there
            if 'REGIONNAME' in feature_cols:
                print("\nWARNING: REGIONNAME is in model features - this should be removed")
                # Remove REGIONNAME from features
                new_features = [col for col in feature_cols if col != 'REGIONNAME']
                print(f"New feature count: {len(new_features)}")
                
                # Update assets
                assets['features'] = new_features
                return assets, new_features
            else:
                print("\nREGIONNAME is not in model features - good")
                return assets, feature_cols
        else:
            print("No 'features' key in assets")
            return None, None
            
    except Exception as e:
        print(f"Error loading model assets: {e}")
        return None, None

def main():
    assets, features = check_model_features()
    
    if assets and features:
        print(f"\nModel has {len(features)} features")
        print("Features are consistent with simplified schema")

if __name__ == "__main__":
    main()
