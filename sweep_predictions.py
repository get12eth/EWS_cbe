import json
import pandas as pd
import numpy as np
import main
from sklearn.metrics import classification_report

DATA_PATH = 'models/Loan_cbe.xlsx'
OUTPUT_PATH = 'sweep_results.json'
NROWS = 5000

def load_data(path=DATA_PATH, nrows=NROWS):
    try:
        df = pd.read_excel(path, engine='openpyxl', nrows=nrows)
        return df
    except Exception as e:
        print('Failed to read dataset:', e)
        return pd.DataFrame()


def run_sweep():
    print('Loading data...')
    df = load_data()
    if df.empty:
        print('No data')
        return

    # detect label column
    label_cols = ['LOAN_STATUS', 'LOAN_STATUS_CLEAN', 'STATUS', 'loan_status', 'LOANSTATUS']
    label_col = None
    for c in label_cols:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]

    df['label_raw'] = df[label_col].astype(str).str.strip().str.upper()
    known = set([str(x).upper() for x in main.label_encoder.classes_])
    df = df[df['label_raw'].isin(known)].reset_index(drop=True)
    if df.empty:
        print('No labeled rows matching model classes')
        return

    X_rows = []
    y_true = []
    valid_idx = []
    for i, r in df.iterrows():
        try:
            arr = main.prepare_model_input(pd.DataFrame([r]))
            if hasattr(arr, 'shape') and arr.shape[0] == 1:
                X_rows.append(arr[0])
                y_true.append(r['label_raw'])
                valid_idx.append(i)
        except Exception:
            continue

    if len(X_rows) == 0:
        print('No valid prepared rows')
        return

    X = np.vstack(X_rows)
    print('Prepared X shape:', X.shape)

    # Predictions
    preds_encoded = main.model.predict(X)
    preds_before = main.label_encoder.inverse_transform(preds_encoded)

    probs = None
    if hasattr(main.model, 'predict_proba'):
        probs = main.model.predict_proba(X)

    # Apply thresholds
    thresholds = getattr(main, 'CLASS_THRESHOLDS', {'NPL':0.01,'SME':0.03,'SET':0.15})
    preds_after = []
    promotions = 0
    promotions_by_class = {}
    for i_idx in range(len(preds_before)):
        before = preds_before[i_idx]
        after = before
        if probs is not None:
            p = probs[i_idx]
            cls_probs = {str(c): float(p[j]) if j < len(p) else 0.0 for j,c in enumerate(main.label_encoder.classes_)}
            candidates = [(cls, prob) for cls, prob in cls_probs.items() if cls != 'PAS' and prob >= thresholds.get(cls, 1.0)]
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                after = candidates[0][0]
        preds_after.append(after)
        if after != before:
            promotions += 1
            promotions_by_class[after] = promotions_by_class.get(after, 0) + 1

    # Metrics
    y_true_arr = np.array(y_true)
    preds_before_arr = np.array(preds_before)
    preds_after_arr = np.array(preds_after)

    report_before = classification_report(y_true_arr, preds_before_arr, output_dict=True, zero_division=0)
    report_after = classification_report(y_true_arr, preds_after_arr, output_dict=True, zero_division=0)

    summary = {
        'rows_evaluated': int(len(y_true_arr)),
        'promotions_total': int(promotions),
        'promotions_by_class': promotions_by_class,
        'thresholds_used': thresholds,
        'report_before': report_before,
        'report_after': report_after
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    print('Sweep complete. Rows:', summary['rows_evaluated'], 'Promotions:', summary['promotions_total'])
    for k,v in promotions_by_class.items():
        print('Promoted to', k, ':', v)


if __name__ == '__main__':
    run_sweep()
