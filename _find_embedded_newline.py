p = r'E:\poridhi\DA-Lab-Project\data\raw\application_train.csv'
# parse with csv but track when a row consumed more than one source line
with open(p, 'r', encoding='utf-8', newline='') as f:
    # use the underlying file iteration, counting physical newlines per logical row
    import csv
    text = f.read()
lines = text.split('\n')
# but csv.reader handles quotes — count how many physical lines are merged per logical row
# Re-open and iterate with csv to find merged lines
import io, csv
reader = csv.reader(io.StringIO(text))
header = next(reader)
expected_cols = len(header)
# track how many physical newlines each logical row consumed
with open(p, 'rb') as f:
    raw = f.read().decode('utf-8')
# Each csv row consumes at least one newline. Walk through and find rows with embedded \n.
# Trick: count '\n' in header line vs data rows.
# Header is the first physical line: from byte 0 up to first '\n'.
hdr_end = raw.index('\n')
print('header_line_len:', hdr_end, 'header_fields:', expected_cols)
header_line = raw[:hdr_end]
print('commas_in_header:', header_line.count(','))
# After header, scan for any logical row containing a '\n' in a quoted field.
# Use csv with strict=True and tracking
reader = csv.reader(io.StringIO(raw))
rows = list(reader)
print('csv_total_rows:', len(rows), 'expected_data_rows:', len(rows)-1)
# Now find row with embedded newline: count '\n' inside its parsed fields
embedded = []
for i, row in enumerate(rows):
    for field in row:
        if '\n' in field:
            embedded.append((i, field))
            break
print('rows with embedded newline:', len(embedded))
for idx, field in embedded[:10]:
    print(f'row {idx} embedded field: {field!r}')