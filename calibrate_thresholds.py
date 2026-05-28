import json
import numpy as np
import pandas as pd
import asyncio
import sys

# Import model and helpers from main
import main
from main import model, label_encoder, prepare_model_input, feature_cols

OUTPUT_PATH = 'calibrated_thresholds.json'

def load_data(path='models/Loan_cbe.xlsx'):
    try:
        df = pd.read_excel(path, engine='openpyxl')
        return df
    except Exception as e:
        print('Failed to read dataset:', e)
        sys.exit(1)


def get_clean_df(df):
    # Try common label column names
    label_cols = ['LOAN_STATUS', 'LOAN_STATUS_CLEAN', 'STATUS', 'loan_status', 'LOANSTATUS']
    label_col = None
    for c in label_cols:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        # fallback to the last column which may be status
        label_col = df.columns[-1]
        print('Assuming label column:', label_col)

    df = df.copy()
    df['label_raw'] = df[label_col].astype(str).str.strip().str.upper()
    # Keep only rows with classes known to label_encoder
    known = set([str(x).upper() for x in label_encoder.classes_])
    df = df[df['label_raw'].isin(known)].reset_index(drop=True)
    if df.empty:
        print('No matching labeled rows found using label_encoder classes:', label_encoder.classes_)
        sys.exit(1)
    return df, label_col


def prepare_inputs(df):
    # The prepare_model_input expects a DataFrame with original columns
    X_arr = prepare_model_input(df)
    return X_arr


def evaluate_thresholds(y_true_encoded, probs, classes, steps=1001):
    # probs: n x k, classes: list of class names
    results = {}
    thresholds = np.linspace(0.0, 1.0, steps)
    for cls in classes:
        idx = list(label_encoder.classes_).index(cls)
        scores = probs[:, idx]
        best = {'threshold': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        for t in thresholds:
            preds = (scores >= t).astype(int)
            y_true = (y_true_encoded == idx).astype(int)
            tp = int(((preds == 1) & (y_true == 1)).sum())
            fp = int(((preds == 1) & (y_true == 0)).sum())
            fn = int(((preds == 0) & (y_true == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best['f1']:
                best = {'threshold': float(t), 'f1': float(f1), 'precision': float(prec), 'recall': float(rec)}
        results[cls] = best
    return results


def main_calibrate():
    print('Loading dataset...')
    df = load_data()
    df_clean, label_col = get_clean_df(df)
    print('Using label column:', label_col, 'Rows:', len(df_clean))

    # Use a holdout split
    from sklearn.model_selection import train_test_split
    train_df, holdout_df = train_test_split(df_clean, test_size=0.25, random_state=42, shuffle=True, stratify=df_clean['label_raw'])
    print('Holdout size:', len(holdout_df))

    # Prepare inputs for holdout
    X_hold = prepare_inputs(holdout_df)
    print('Prepared holdout X shape:', X_hold.shape)

    # Get probabilities
    if not hasattr(model, 'predict_proba'):
        print('Model does not support predict_proba')
        sys.exit(1)
    probs = model.predict_proba(X_hold)

    # Map holdout labels to encoder indices
    # label_encoder.classes_ are likely like ['NPL','PAS','SET','SME'] but check
    # Our holdout labels are uppercase strings
    name_to_idx = {str(c).upper(): i for i, c in enumerate(label_encoder.classes_)}
    y_true_names = holdout_df['label_raw'].tolist()
    y_true_encoded = np.array([name_to_idx.get(n, -1) for n in y_true_names])
    valid_mask = y_true_encoded >= 0

    if not valid_mask.all():
        print('Warning: some labels in holdout are unknown to encoder; dropping')
        probs = probs[valid_mask]
        y_true_encoded = y_true_encoded[valid_mask]

    classes_of_interest = [c for c in label_encoder.classes_ if c in ['NPL','SME','SET']]
    print('Classes found in encoder:', list(label_encoder.classes_))
    print('Calibrating for:', classes_of_interest)

    results = evaluate_thresholds(y_true_encoded, probs, classes_of_interest)

    print('\nCalibration results (best F1 per class):')
    for cls, res in results.items():
        print(f"{cls}: threshold={res['threshold']:.4f}, f1={res['f1']:.4f}, prec={res['precision']:.4f}, rec={res['recall']:.4f}")

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved calibration to', OUTPUT_PATH)


if __name__ == '__main__':
    main_calibrate()
