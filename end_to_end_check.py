import json
import pandas as pd
import numpy as np
import main

DATA_PATH = 'models/Loan_cbe.xlsx'
N = 10

def load_sample(path=DATA_PATH, nrows=2000, sample_n=10):
    try:
        df = pd.read_excel(path, engine='openpyxl', nrows=nrows)
    except Exception as e:
        print('Failed to read dataset:', e)
        return pd.DataFrame()

    # find a label column
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
        print('No matching labeled rows found')
        return pd.DataFrame()

    return df.head(sample_n)


def run_check():
    df = load_sample(sample_n=N)
    if df.empty:
        return

    for i, row in df.iterrows():
        row_df = pd.DataFrame([row])
        X = main.prepare_model_input(row_df)

        # predict
        pred_encoded = main.model.predict(X)[0]
        pred_before = main.label_encoder.inverse_transform([pred_encoded])[0]

        probs = {}
        if hasattr(main.model, 'predict_proba'):
            p = main.model.predict_proba(X)[0]
            for idx, cls in enumerate(main.label_encoder.classes_):
                if idx < len(p):
                    probs[cls] = float(p[idx])
        else:
            probs = {cls: (1.0 if cls == pred_before else 0.0) for cls in main.label_encoder.classes_}

        # apply thresholds
        thresholds = main.CLASS_THRESHOLDS if hasattr(main, 'CLASS_THRESHOLDS') else {'NPL':0.2,'SME':0.15,'SET':0.15}
        candidates = [(cls, prob) for cls, prob in probs.items() if cls != 'PAS' and prob >= thresholds.get(cls, 1.0)]
        pred_after = pred_before
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            pred_after = candidates[0][0]

        print('---')
        print('Index:', i)
        print('Actual label:', row.get('label_raw'))
        print('Predicted before:', pred_before)
        print('Predicted after :', pred_after)
        print('Top probs:', sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])


if __name__ == '__main__':
    run_check()
