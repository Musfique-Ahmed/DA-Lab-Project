import csv, io
p = r'E:\poridhi\DA-Lab-Project\data\raw\application_train.csv'
# Parse with strict csv to detect quoted-newline rows
with open(p, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    header_row = next(reader)
    expected = len(header_row)
    print('header cols:', expected)
    bad = []
    for i, row in enumerate(reader, start=2):
        if len(row) != expected:
            bad.append((i, len(row), row[:3]))
        if len(bad) >= 3:
            break
    print('first bad rows (line_num, ncols, first 3 fields):')
    for b in bad:
        print(b)
# Also test: any row that contains a literal '\n' inside a quoted field?
with open(p, 'rb') as f:
    data = f.read()
# Look for a quoted newline
import re
# find positions of unescaped newlines inside quotes
# Quick check: count quote chars vs escaped quotes
print('total double-quotes:', data.count(b'"'))
print('total even quotes? ', data.count(b'"') % 2 == 0)