#!/usr/bin/env python3
"""
Fix model assets by removing REGIONNAME from features
"""

import joblib
import os

def fix_model_assets():
    """Remove REGIONNAME from model features and save corrected version"""
    try:
        # Load original assets
        assets = joblib.load("models/loan_prediction_assets_1.pkl")
        
        # Get original features
        original_features = assets['features']
        print(f"Original features: {len(original_features)}")
        
        # Remove REGIONNAME from features
        new_features = [col for col in original_features if col != 'REGIONNAME']
        print(f"New features: {len(new_features)}")
        
        # Update assets
        assets['features'] = new_features
        
        # Save corrected assets
        backup_path = "models/loan_prediction_assets_1_backup.pkl"
        corrected_path = "models/loan_prediction_assets_1_fixed.pkl"
        
        # Backup original
        joblib.dump(assets, backup_path)
        print(f"Original assets backed up to: {backup_path}")
        
        # Save corrected version
        joblib.dump(assets, corrected_path)
        print(f"Corrected assets saved to: {corrected_path}")
        
        # Show removed feature
        removed = [col for col in original_features if col not in new_features]
        print(f"Removed features: {removed}")
        
        return True
        
    except Exception as e:
        print(f"Error fixing model assets: {e}")
        return False

def main():
    success = fix_model_assets()
    if success:
        print("\nModel assets fixed successfully!")
        print("Next step: Update main.py to use the corrected assets file")
    else:
        print("\nFailed to fix model assets")

if __name__ == "__main__":
    main()
