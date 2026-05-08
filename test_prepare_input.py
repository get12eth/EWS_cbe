import main
import pandas as pd
import json

row = {}
for c in main.feature_cols:
    if any(k in c.upper() for k in ['BRANCH','REGION','DISTRICT','PRODUCT','LOAN_TYPE','LTYPE','OWNERSHIP','INDUSTRY','SECTOR']):
        row[c] = 'ADDIS ABABA'
    else:
        row[c] = 0

df = pd.DataFrame([row])
arr = main.prepare_model_input(df)
print('Prepared shape:', arr.shape)
print('Row sample:', arr[0][:5])
print('CLASS_THRESHOLDS:', json.dumps(main.CLASS_THRESHOLDS))
