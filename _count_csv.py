import pandas as pd
df = pd.read_csv(r'E:\poridhi\DA-Lab-Project\data\raw\application_train.csv')
ids = sorted(df['SK_ID_CURR'].tolist())
gaps = []
for i in range(len(ids)-1):
    if ids[i+1] - ids[i] != 1:
        gaps.append((ids[i], ids[i+1]))
print('Total gaps:', len(gaps))
print('First 10 gaps:', gaps[:10])
print('Last 5 gaps:', gaps[-5:])
expected_min, expected_max = 100002, 456254
print(f'Range {expected_min}..{expected_max}, count={expected_max-expected_min+1}, missing={expected_max-expected_min+1-len(ids)}')
# Check missing from full range
full = set(range(expected_min, expected_max+1))
missing = sorted(full - set(ids))
print('Missing IDs total:', len(missing))
print('Missing first 30:', missing[:30])
print('Missing last 30:', missing[-30:])