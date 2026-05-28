#!/usr/bin/env python3
"""
Check model classes and label encoder to understand why SME is not predicted
"""

import joblib
import numpy as np

def check_model_classes():
    """Check what classes the model was trained with"""
    try:
        # Load the model assets
        assets = joblib.load('models/loan_prediction_assets_1.pkl')
        
        # Check the label encoder classes
        label_encoder = assets['label_encoder']
        print('Label Encoder Classes:')
        for i, class_name in enumerate(label_encoder.classes_):
            print(f'  {i}: {class_name}')
        print(f'Number of classes: {len(label_encoder.classes_)}')
        
        # Check the model type and structure
        model = assets['model']
        print(f'\nModel type: {type(model)}')
        
        # For XGBoost models
        if hasattr(model, 'n_classes_'):
            print(f'Model n_classes_: {model.n_classes_}')
        elif hasattr(model, 'num_class'):
            print(f'Model num_class: {model.num_class}')
        
        # Check feature columns
        features = assets['features']
        print(f'\nNumber of features: {len(features)}')
        print('First 10 features:')
        for i, feature in enumerate(features[:10]):
            print(f'  {i}: {feature}')
            
        return label_encoder.classes_
        
    except Exception as e:
        print(f'Error checking model classes: {e}')
        return None

def check_prediction_output():
    """Check what the model actually outputs"""
    try:
        # Load model assets
        assets = joblib.load('models/loan_prediction_assets_1.pkl')
        model = assets['model']
        label_encoder = assets['label_encoder']
        feature_cols = assets['features']
        
        # Create dummy input data
        dummy_input = np.random.rand(1, len(feature_cols))
        
        # Get prediction
        pred = model.predict(dummy_input)
        pred_proba = model.predict_proba(dummy_input)
        
        print(f'\nPrediction output: {pred}')
        print(f'Prediction probabilities: {pred_proba}')
        print(f'Predicted class index: {pred[0]}')
        
        if pred[0] < len(label_encoder.classes_):
            predicted_class = label_encoder.classes_[pred[0]]
            print(f'Predicted class name: {predicted_class}')
        else:
            print(f'Prediction index {pred[0]} is out of range for {len(label_encoder.classes_)} classes')
            
        return pred, pred_proba
        
    except Exception as e:
        print(f'Error checking prediction output: {e}')
        return None, None

def main():
    """Main function"""
    print("Model Classes and Prediction Check")
    print("=" * 50)
    
    classes = check_model_classes()
    pred, pred_proba = check_prediction_output()
    
    print("\n" + "=" * 50)
    print("Analysis:")
    if classes is not None:
        print(f"Model was trained with {len(classes)} classes: {list(classes)}")
        print(f"Available classes: {', '.join(classes)}")
        
        if 'SME' not in classes:
            print("⚠️  SME is NOT in the training classes - this explains why it's never predicted!")
        else:
            print("✅ SME is in the training classes")
            
        if 'SET' not in classes:
            print("⚠️  SET is NOT in the training classes")
        else:
            print("✅ SET is in the training classes")

if __name__ == "__main__":
    main()
