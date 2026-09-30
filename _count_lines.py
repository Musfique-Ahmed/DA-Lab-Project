import pandas as pd, os
p = r'E:\poridhi\DA-Lab-Project\data\raw\application_train.csv'
print('size_bytes:', os.path.getsize(p))
# count via pandas (high-level csv): same as loader does
df = pd.read_csv(p)
print('pandas_shape:', df.shape)
# count lines via chunked read (no overhead): sum row counts across chunks
total = 0
for chunk in pd.read_csv(p, chunksize=50000):
    total += len(chunk)
print('chunked_total:', total)
# count newlines in raw file
with open(p, 'rb') as f:
    data = f.read()
print('raw_newlines:', data.count(b'\n'))
print('raw_size:', len(data))