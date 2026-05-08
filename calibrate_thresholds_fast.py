import json
import numpy as np
import pandas as pd
import sys

# Import model and helpers from main
import main
from main import model, label_encoder, prepare_model_input

OUTPUT_PATH = 'calibrated_thresholds_fast.json'

def load_data(path='models/Loan_cbe.xlsx', nrows=2000):
    try:
        df = pd.read_excel(path, engine='openpyxl', nrows=nrows)
        return df
    except Exception as e:
        print('Failed to read dataset:', e)
        sys.exit(1)


def main_calibrate(nrows=20000):
    print('Loading dataset (fast mode)...')
    df = load_data(nrows=nrows)

    # Identify label column
    label_cols = ['LOAN_STATUS', 'LOAN_STATUS_CLEAN', 'STATUS', 'loan_status', 'LOANSTATUS']
    label_col = None
    for c in label_cols:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]
        print('Assuming label column:', label_col)

    df['label_raw'] = df[label_col].astype(str).str.strip().str.upper()
    known = set([str(x).upper() for x in label_encoder.classes_])
    df = df[df['label_raw'].isin(known)].reset_index(drop=True)
    if df.empty:
        print('No matching labeled rows found')
        sys.exit(1)

    from sklearn.model_selection import train_test_split
    try:
        train_df, holdout_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df['label_raw'])
    except ValueError:
        # Fallback to random split when stratify isn't possible due to very small classes
        train_df, holdout_df = train_test_split(df, test_size=0.25, random_state=42, shuffle=True)
    print('Holdout size:', len(holdout_df))

    # Prepare inputs row-by-row to isolate encoding errors and ensure shapes match
    X_rows = []
    valid_indices = []
    for i, r in holdout_df.iterrows():
        try:
            arr = prepare_model_input(pd.DataFrame([r]))
            # ensure we have a 1D row
            if hasattr(arr, 'shape') and arr.shape[0] == 1:
                X_rows.append(arr[0])
                valid_indices.append(i)
            else:
                # unexpected shape, skip
                continue
        except Exception:
            # skip problematic rows
            continue

    if len(X_rows) == 0:
        print('No valid prepared rows in holdout')
        sys.exit(1)

    X_hold = np.vstack(X_rows)
    print('Prepared X shape:', getattr(X_hold, 'shape', None))

    if not hasattr(model, 'predict_proba'):
        print('Model does not support predict_proba')
        sys.exit(1)
    probs = model.predict_proba(X_hold)

    name_to_idx = {str(c).upper(): i for i, c in enumerate(label_encoder.classes_)}
    y_true_names = holdout_df['label_raw'].tolist()
    y_true_encoded = np.array([name_to_idx.get(n, -1) for n in y_true_names])
    valid_mask = y_true_encoded >= 0
    if not valid_mask.all():
        probs = probs[valid_mask]
        y_true_encoded = y_true_encoded[valid_mask]

    classes_of_interest = [c for c in label_encoder.classes_ if c in ['NPL','SME','SET']]
    print('Calibrating for:', classes_of_interest)

    results = {}
    thresholds = np.linspace(0.0, 1.0, 101)
    for cls in classes_of_interest:
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

    print('\nCalibration results:')
    for cls, res in results.items():
        print(f"{cls}: threshold={res['threshold']:.4f}, f1={res['f1']:.4f}, prec={res['precision']:.4f}, rec={res['recall']:.4f}")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved calibration to', OUTPUT_PATH)


if __name__ == '__main__':
    main_calibrate()
